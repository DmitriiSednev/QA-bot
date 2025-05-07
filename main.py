import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta

# import io # Больше не нужен здесь
import easyocr  # Оставляем для инициализации

# import signal # Убираем импорт signal

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
)
from telegram.constants import ParseMode

# --- Import Agent Logic ---
from agent.graph_builder import setup_agent

# --- База данных и модели (если нужны напрямую, например, для первоначальной проверки админов) ---
from database import connection, models
from database.crud_admin import get_admin_by_user_id, add_admin
from database.redis_cache import RedisCache  # Импортируем класс кеша

# --- Импорт обработчиков и задач ---
from telegram_interface import handlers, jobs

# --- Импорт утилит блокировки ---
from core.lock import acquire_lock, release_lock

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
BOT_USERNAME = os.getenv("BOT_USERNAME")  # Имя пользователя бота (без @)

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

# --- Инициализация OCR ---
OCR_READER = None
try:
    # Указываем директорию для моделей, если нужно
    # model_storage_directory = os.path.join(os.getcwd(), '.model_cache')
    # os.makedirs(model_storage_directory, exist_ok=True)
    OCR_READER = easyocr.Reader(
        ["ru", "en"],
        gpu=False,  # Установить в True, если есть GPU и нужные библиотеки
        # model_storage_directory=model_storage_directory,
        # download_enabled=True # Разрешить скачивание моделей
    )
    logger.info("OCR Reader (easyocr) инициализирован для языков [ru, en].")
except Exception as e:
    logger.error(
        f"Ошибка инициализации OCR Reader: {e}. Обработка изображений будет недоступна.",
        exc_info=True,
    )
    OCR_READER = None  # Убедимся, что None если ошибка

# --- Инициализация Агента LangGraph ---
agent_app = None
checkpoint_db_connection = None  # Для хранения соединения с БД чекпоинтера


# Обертка для асинхронной инициализации агента
async def initialize_bot_resources():
    """Асинхронно инициализирует все ресурсы бота, включая агент."""
    global agent_app, checkpoint_db_connection, redis_cache_instance, OCR_READER

    # Инициализация Redis
    redis_cache_instance = None
    try:
        redis_cache_instance = RedisCache()
        if await asyncio.to_thread(
            redis_cache_instance.ping
        ):  # Выполняем ping в потоке
            logger.info(
                f"Успешное подключение к Redis: {os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}/{os.getenv('REDIS_DB')}"
            )
        else:
            logger.error(
                "Не удалось подключиться к Redis. Кеширование будет отключено."
            )
            redis_cache_instance = None
    except Exception as e:
        logger.error(f"Ошибка при инициализации Redis кэша: {e}", exc_info=True)
        redis_cache_instance = None

    # Инициализация OCR
    OCR_READER = None
    try:
        OCR_READER = await asyncio.to_thread(easyocr.Reader, ["ru", "en"], gpu=False)
        logger.info("OCR Reader (easyocr) инициализирован для языков [ru, en].")
    except Exception as e:
        logger.error(
            f"Ошибка инициализации OCR Reader: {e}. Обработка изображений будет недоступна.",
            exc_info=True,
        )
        OCR_READER = None

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
            if agent_app_tuple and agent_app_tuple[1]:
                await agent_app_tuple[1].close()
            sys.exit(1)
    except Exception as e:
        logger.critical(
            f"Критическая ошибка при асинхронной инициализации агента: {e}",
            exc_info=True,
        )
        if checkpoint_db_connection:
            try:
                await checkpoint_db_connection.close()
            except Exception as close_e:
                logger.error(
                    f"Ошибка при закрытии checkpoint_db_connection после сбоя инициализации: {close_e}"
                )
        sys.exit(1)


# --- Основная функция запуска бота ---
async def main() -> None:
    """Инициализирует ресурсы, настраивает и запускает бота."""
    # 0. Захват блокировки файла (до инициализации ресурсов)
    if not acquire_lock():
        sys.exit(1)

    application = None  # Определяем application здесь для finally
    try:
        # 1. Асинхронная инициализация всех ресурсов
        logger.info("Инициализация ресурсов бота...")
        await initialize_bot_resources()
        logger.info("Ресурсы бота успешно инициализированы.")

        # 2. Гарантируем наличие первого админа в БД (синхронная операция)
        ensure_initial_admin()

        # 3. Настройка приложения Telegram
        logger.info("Настройка приложения Telegram...")
        defaults = Defaults(parse_mode=ParseMode.MARKDOWN)
        application = (
            ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).defaults(defaults).build()
        )

        # 4. Сохранение важных данных в bot_data
        application.bot_data["agent_app"] = agent_app
        application.bot_data["ocr_reader"] = OCR_READER
        application.bot_data["admin_ids"] = ADMIN_IDS
        application.bot_data["bot_username"] = BOT_USERNAME
        application.bot_data["redis_cache"] = redis_cache_instance
        application.bot_data["days_to_keep_faq"] = int(
            os.getenv("DAYS_TO_KEEP_FAQ", "365")
        )
        application.bot_data["days_to_keep_history"] = int(
            os.getenv("DAYS_TO_KEEP_CHAT_HISTORY", "30")
        )

        # 5. Регистрация обработчиков
        handlers.setup_handlers(
            application
        )  # Передаем application в функцию настройки хендлеров
        logger.info("Обработчики команд и сообщений зарегистрированы.")

        # 6. Настройка и запуск планировщика задач
        jobs.setup_jobs(application)  # Передаем application в функцию настройки задач
        logger.info("Планировщик задач настроен.")

        # 7. Установка команд бота
        async def set_commands(app: Application):
            try:
                await app.bot.set_my_commands(handlers.DEFAULT_COMMANDS)
                logger.info("Команды бота успешно установлены.")
            except Exception as e:
                logger.error(f"Ошибка при установке команд бота: {e}", exc_info=True)

        application.post_init = set_commands

        # 8. Запуск бота
        logger.info("Запуск бота...")
        await application.initialize()  # Инициализируем приложение
        await application.start()  # Запускаем внутренние компоненты
        await application.updater.start_polling(
            allowed_updates=Update.ALL_TYPES
        )  # Начинаем опрос
        logger.info("Бот успешно запущен и работает.")

        # Поддерживаем работу скрипта (ожидание сигнала остановки)
        # await application.updater.idle() # Этот метод блокирует и ждет сигналы
        # Или просто бесконечный цикл
        while True:
            await asyncio.sleep(3600)  # Проверка каждые N секунд

    except (KeyboardInterrupt, SystemExit) as e:
        logger.warning(
            f"Получен сигнал остановки ({type(e).__name__}). Завершение работы..."
        )
        # Логика остановки будет в finally
    except Exception as e:
        logger.critical(f"Критическая ошибка во время работы бота: {e}", exc_info=True)
    finally:
        logger.info("Начало процедуры остановки бота...")
        if application and application.updater:
            if application.updater.running:
                logger.info("Остановка Polling...")
                await application.updater.stop()
            else:
                logger.info("Polling уже был остановлен.")
        else:
            logger.warning(
                "Application или Updater не инициализированы для остановки polling."
            )

        if application:
            logger.info("Остановка Application...")
            await application.stop()
            await application.shutdown()
            logger.info("Application остановлен.")
        else:
            logger.warning(
                "Application не был инициализирован для вызова stop/shutdown."
            )

        logger.info("Закрытие соединения с БД чекпоинтера...")
        if checkpoint_db_connection:
            try:
                await checkpoint_db_connection.close()
                logger.info("Соединение aiosqlite для checkpointer успешно закрыто.")
            except Exception as e:
                logger.error(
                    f"Ошибка при закрытии соединения aiosqlite для checkpointer: {e}",
                    exc_info=True,
                )
        else:
            logger.info("Соединение aiosqlite не было открыто или уже закрыто.")

        logger.info("Освобождение блокировки файла...")
        release_lock()
        logger.info("Все ресурсы освобождены. Выход.")


if __name__ == "__main__":
    asyncio.run(main())
