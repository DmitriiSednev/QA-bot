import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
import signal  # <--- ДОБАВЛЯЕМ ИМПОРТ
import functools  # <--- ДОБАВЛЯЕМ ИМПОРТ

# import io # Больше не нужен здесь
# import easyocr # УДАЛЯЕМ ИМПОРТ EASYOCR ОТСЮДА, он будет в handlers

# import portalocker # Удаляем, т.к. логика в core.lock
from typing import Optional, Dict, Set  # Any, List, Union могут быть не нужны

from dotenv import load_dotenv
from telegram import Update, BotCommand
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    Defaults,
    ApplicationBuilder,
    CallbackQueryHandler,  # Добавим, если будут инлайн-кнопки
    AIORateLimiter,  # Для ограничения частоты запросов
)
from telegram.constants import ParseMode

# --- Import Agent Logic ---
from agent.graph_builder import (
    build_graph,  # Оставляем build_graph, если он где-то нужен, но setup_agent важнее
    AgentState,
)
from agent import setup_agent  # <--- ИЗМЕНЕНИЕ ЗДЕСЬ
from agent.llm_setup import (
    setup_llm,
)  # Для инициализации LLM (хотя build_graph это делает)
from database.connection import get_db_session  # Оставляем только нужный импорт
from database.redis_cache import (
    RedisCache,
    close_redis_connection,
)  # Закрытие Redis и импорт класса
from core.lock import (
    acquire_lock,
    release_lock,
    register_signal_handlers,
)  # ВОЗВРАЩАЕМ ЭТИ ИМПОРТЫ
from telegram_interface.handlers import (
    start,
    # help_command, # УДАЛЕНО
    # 개발자_정보_command,  # УДАЛЯЕМ ЭТОТ ИМПОРТ
    handle_message,  # Основной обработчик сообщений
    handle_photo,  # Обработчик изображений (OCR)
    handle_new_chat_members,  # Обработчик добавления в чат
    admin_command,  # Обработчик админ-команд
)
from telegram_interface.jobs import setup_jobs  # Настройка периодических задач

# --- База данных и модели (если нужны напрямую, например, для первоначальной проверки админов) ---
from database import connection, models
from database.crud_admin import get_admin_by_user_id, add_admin

# --- Импорт обработчиков и задач ---
from telegram_interface import handlers, jobs

# --- Конфигурация логирования ---
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "bot.log")
# LOCK_FILE перенесен в core.lock

# Создаем директорию логов, если ее нет
os.makedirs(LOG_DIR, exist_ok=True)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE)],
)
logging.getLogger("httpx").setLevel(
    logging.WARNING
)  # Убираем излишнее логирование HTTPX
logging.getLogger("easyocr").setLevel(
    logging.ERROR
)  # Убираем детальное логирование EasyOCR
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(
    logging.WARNING
)  # Уменьшаем шум от apscheduler

logger = logging.getLogger(__name__)

# --- Загрузка переменных окружения ---
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_USER_IDS_STR = os.getenv("ADMIN_USER_IDS")
BOT_USERNAME = os.getenv("TELEGRAM_BOT_USERNAME")  # Имя пользователя бота (без @)

# --- Обработка ID администраторов ---
ADMIN_IDS: Set[int] = set()
if ADMIN_USER_IDS_STR:
    try:
        ADMIN_IDS = {
            int(admin_id.strip())
            for admin_id in ADMIN_USER_IDS_STR.split(",")
            if admin_id.strip()
        }
        logger.info(f"Загружены ID администраторов из .env: {ADMIN_IDS}")
    except ValueError:
        logger.error(
            "Ошибка парсинга ADMIN_USER_IDS в .env. Убедитесь, что это числа через запятую."
        )
        ADMIN_IDS = set()

# --- Проверка критических переменных ---
if not TELEGRAM_BOT_TOKEN:
    logger.critical("TELEGRAM_BOT_TOKEN не найден! Бот не может запуститься.")
    sys.exit(1)
if not BOT_USERNAME:
    logger.warning(
        "BOT_USERNAME не найден в .env. Некоторые функции (ответы в группах) могут работать некорректно."
    )


# Добавляем первого админа из .env в БД при первом запуске, если его там нет
def ensure_initial_admin():
    if not ADMIN_IDS:
        logger.warning(
            "Нет ADMIN_USER_IDS в .env, не могу добавить первоначального админа."
        )
        return

    first_admin_id = next(iter(ADMIN_IDS))  # Берем первого админа из списка
    try:
        with connection.get_db_session() as db:
            if db:
                admin = get_admin_by_user_id(db, first_admin_id)
                if not admin:
                    logger.info(
                        f"Первый админ (ID: {first_admin_id}) не найден в БД. Добавляю..."
                    )
                    # Пытаемся добавить без username, т.к. его может не быть при старте
                    add_admin(db, first_admin_id, username=None)
                else:
                    logger.info(
                        f"Первый админ (ID: {first_admin_id}) уже существует в БД."
                    )
            else:
                logger.error(
                    "Не удалось получить сессию БД для проверки/добавления админа."
                )
    except Exception as e:
        logger.error(
            f"Ошибка при добавлении первоначального админа в БД: {e}", exc_info=True
        )


# --- Инициализация Redis Cache ---
redis_cache_instance = None
try:
    # Инициализируем RedisCache без URL, он сам возьмет переменные окружения
    redis_cache_instance = RedisCache()
    # Проверка соединения
    if redis_cache_instance.ping():
        logger.info(
            f"Успешное подключение к Redis: {os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}/{os.getenv('REDIS_DB')}"
        )
    else:
        logger.error("Не удалось подключиться к Redis. Кеширование будет отключено.")
        redis_cache_instance = None
except Exception as e:
    logger.error(f"Ошибка при инициализации Redis кэша: {e}", exc_info=True)
    redis_cache_instance = None

# --- Инициализация OCR (УДАЛЕНО ОТСЮДА) ---
# OCR_READER = None

# --- Инициализация Агента LangGraph ---
agent_app = None
checkpoint_db_connection = None  # Для хранения соединения с БД чекпоинтера


# Обертка для асинхронной инициализации агента
async def initialize_bot_resources(application: Application):
    """Асинхронно инициализирует все ресурсы бота, кроме запуска самого приложения Telegram."""
    global agent_app, checkpoint_db_connection, redis_cache_instance

    # Инициализация Redis (ПЕРЕМЕЩЕНО ВЫШЕ, ВНЕ initialize_bot_resources)
    # redis_cache_instance = None  # Сбрасываем, если была предыдущая попытка
    # try:
    #     redis_cache_instance = RedisCache()
    #     if await asyncio.to_thread(
    #         redis_cache_instance.ping
    #     ):  # Выполняем ping в потоке
    #         logger.info(
    #             f"Успешное подключение к Redis: {os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}/{os.getenv('REDIS_DB')}"
    #         )
    #     else:
    #         logger.error(
    #             "Не удалось подключиться к Redis. Кеширование будет отключено."
    #         )
    #         redis_cache_instance = None
    # except Exception as e:
    #     logger.error(f"Ошибка при инициализации Redis кэша: {e}", exc_info=True)
    #     redis_cache_instance = None

    # Инициализация Агента LangGraph
    try:
        agent_app_tuple = await setup_agent()
        if (
            agent_app_tuple
            and agent_app_tuple[0] is not None
            and agent_app_tuple[1] is not None
        ):
            agent_app, checkpoint_db_connection = agent_app_tuple
            logger.info("Агент и соединение с БД чекпоинтера успешно инициализированы.")
        else:
            logger.critical(
                "Не удалось настроить приложение агента или соединение с БД чекпоинтера. Завершение работы."
            )
            if (
                agent_app_tuple and agent_app_tuple[1]
            ):  # Если соединение было создано, но агент нет
                try:
                    await agent_app_tuple[1].close()
                except Exception as close_e:
                    logger.error(
                        f"Ошибка при закрытии checkpoint_db_connection после сбоя инициализации агента: {close_e}"
                    )
            sys.exit(1)  # Критическая ошибка, выходим
    except Exception as e:
        logger.critical(
            f"Критическая ошибка при асинхронной инициализации агента: {e}",
            exc_info=True,
        )
        if checkpoint_db_connection:  # Если соединение было создано до ошибки
            try:
                await checkpoint_db_connection.close()
            except Exception as close_e:
                logger.error(
                    f"Ошибка при закрытии checkpoint_db_connection после сбоя инициализации агента: {close_e}"
                )
        sys.exit(1)  # Критическая ошибка, выходим

    # Сохранение важных данных в bot_data
    application.bot_data["agent_app"] = agent_app
    application.bot_data["ocr_reader"] = (
        None  # Инициализация OCR будет ленивой в обработчике
    )
    application.bot_data["admin_ids"] = ADMIN_IDS
    application.bot_data["bot_username"] = BOT_USERNAME
    application.bot_data["redis_cache"] = redis_cache_instance
    application.bot_data["days_to_keep_faq"] = int(os.getenv("DAYS_TO_KEEP_FAQ", "365"))
    application.bot_data["days_to_keep_chat_history"] = int(
        os.getenv("DAYS_TO_KEEP_CHAT_HISTORY", "30")
    )
    logger.info("Основные ресурсы бота (агент, Redis, bot_data) инициализированы.")


# --- Новая функция для post_shutdown --- #
async def custom_post_shutdown(app: Application):
    """Асинхронные операции при завершении работы бота."""
    logger.info("Начало выполнения custom_post_shutdown...")
    global checkpoint_db_connection
    if checkpoint_db_connection:
        try:
            await checkpoint_db_connection.close()
            logger.info(
                "Соединение aiosqlite для checkpointer успешно закрыто в post_shutdown."
            )
        except Exception as e:
            logger.error(
                f"Ошибка при закрытии соединения aiosqlite для checkpointer в post_shutdown: {e}",
                exc_info=True,
            )
    else:
        logger.info(
            "Соединение aiosqlite (checkpointer) не было открыто или уже закрыто (post_shutdown)."
        )

    # Другие асинхронные операции по очистке, если они есть
    # Например, если redis_cache_instance имел бы async close метод:
    # global redis_cache_instance
    # if redis_cache_instance and hasattr(redis_cache_instance, 'aclose'):
    #     try:
    #         await redis_cache_instance.aclose()
    #         logger.info("Асинхронное соединение Redis успешно закрыто в post_shutdown.")
    #     except Exception as e:
    #         logger.error(f"Ошибка при асинхронном закрытии Redis в post_shutdown: {e}", exc_info=True)
    logger.info("custom_post_shutdown завершен.")


# --- Основная функция запуска бота ---
def main() -> None:
    global application_instance

    if not acquire_lock():
        sys.exit(1)

    application_instance = None

    try:
        logger.info("Настройка приложения Telegram...")
        defaults = Defaults(parse_mode=ParseMode.MARKDOWN)
        application_instance = (
            ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).defaults(defaults).build()
        )
        logger.info("Приложение Telegram настроено.")

        # Асинхронная инициализация через post_init
        async def post_init(app: Application):
            await initialize_bot_resources(app)
            ensure_initial_admin()
            logger.info("Асинхронная post_init инициализация завершена.")

        application_instance.post_init = post_init
        application_instance.post_shutdown = (
            custom_post_shutdown  # <--- ПРИСВАИВАЕМ ФУНКЦИЮ
        )

        handlers.setup_handlers(application_instance)
        logger.info("Обработчики команд и сообщений зарегистрированы.")

        jobs.setup_jobs(application_instance)
        logger.info("Планировщик задач настроен.")

        logger.info("Запуск бота (run_polling)...")
        application_instance.run_polling(allowed_updates=Update.ALL_TYPES)
        logger.info("run_polling завершился (это неожиданно при штатной работе).")

    except Exception as e:
        logger.critical(
            f"MAIN: Unhandled Exception in main try block: {e}", exc_info=True
        )
    finally:
        logger.info("Начало процедуры остановки бота (finally)...")
        # Закрытие checkpoint_db_connection перенесено в custom_post_shutdown
        # logger.info("Закрытие соединения с БД чекпоинтера...")
        # global checkpoint_db_connection
        # if checkpoint_db_connection:
        #     try:
        #         import asyncio
        #         asyncio.run(checkpoint_db_connection.close())
        #         logger.info("Соединение aiosqlite для checkpointer успешно закрыто.")
        #     except Exception as e:
        #         logger.error(
        #             f"Ошибка при закрытии соединения aiosqlite для checkpointer: {e}",
        #             exc_info=True,
        #         )
        # else:
        #     logger.info("Соединение aiosqlite не было открыто или уже закрыто.")

        logger.info("Закрытие соединения с Redis (синхронно)...")
        try:
            close_redis_connection()  # Это синхронная функция
            logger.info("Глобальное соединение Redis успешно закрыто (синхронно).")
        except Exception as e:
            logger.error(
                f"Ошибка при вызове close_redis_connection: {e}", exc_info=True
            )

        logger.info("Освобождение блокировки файла...")
        release_lock()
        logger.info("Все ресурсы освобождены. Выход.")


if __name__ == "__main__":
    # Инициализация Redis (как и было)
    if redis_cache_instance is None:
        logger.info("Переинициализация Redis Cache перед запуском...")
        try:
            redis_cache_instance = RedisCache()
            if redis_cache_instance.ping():
                logger.info(
                    f"Успешное подключение к Redis перед запуском: {os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}/{os.getenv('REDIS_DB')}"
                )
            else:
                logger.error(
                    "Не удалось подключиться к Redis перед запуском. Кеширование будет отключено."
                )
                redis_cache_instance = None
        except Exception as e:
            logger.error(
                f"Ошибка при переинициализации Redis кэша перед запуском: {e}",
                exc_info=True,
            )
            redis_cache_instance = None

    # Просто вызываем main(), НЕ asyncio.run(main())
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Получен сигнал KeyboardInterrupt. Завершение бота...")
    except SystemExit as e:
        logger.warning(f"Получен сигнал SystemExit ({e}). Завершение бота...")
    except Exception as e:
        logger.critical(
            f"Необработанное исключение на верхнем уровне: {e}", exc_info=True
        )
