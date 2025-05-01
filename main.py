import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
import io
import easyocr
import signal
import fcntl
from typing import Optional, Dict, Any, List, Union

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    Defaults,
)

# --- Import Agent Logic ---
from agent.agent_executor import setup_agent
from langgraph.graph.message import AnyMessage
from langchain_core.messages import HumanMessage, AIMessage
from database import connection, crud
from langgraph.graph import END, StateGraph
from langchain.agents.agent import AgentExecutor
from agent.tools.chat_history_loader import ChatHistoryLoaderTool

# --- Конфигурация логирования --- #
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "bot.log")
LOCK_FILE = "/tmp/telegram_bot.lock"

# Создаем директорию логов, если ее нет
os.makedirs(LOG_DIR, exist_ok=True)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)

def acquire_lock():
    """Попытка получить блокировку для предотвращения множественных запусков."""
    try:
        # Проверяем, существует ли файл блокировки
        if os.path.exists(LOCK_FILE):
            # Проверяем, активен ли процесс
            with open(LOCK_FILE, 'r') as f:
                try:
                    old_pid = int(f.read().strip())
                    if os.path.exists(f"/proc/{old_pid}"):
                        logger.error(f"Бот уже запущен (PID: {old_pid})")
                        sys.exit(1)
                    else:
                        # Процесс не существует, удаляем старый файл блокировки
                        os.unlink(LOCK_FILE)
                except (ValueError, FileNotFoundError):
                    # Некорректный PID или процесс уже завершен
                    os.unlink(LOCK_FILE)
        
        # Создаем новый файл блокировки
        with open(LOCK_FILE, 'w') as f:
            f.write(str(os.getpid()))
        
        # Получаем файловый дескриптор для fcntl
        lock_fd = open(LOCK_FILE, 'r')
        fcntl.lockf(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lock_fd
    except IOError:
        logger.error("Не удалось получить блокировку. Возможно, бот уже запущен.")
        sys.exit(1)

def release_lock(lock_fd):
    """Освобождение блокировки."""
    try:
        if lock_fd:
            fcntl.lockf(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
        if os.path.exists(LOCK_FILE):
            os.unlink(LOCK_FILE)
    except Exception as e:
        logger.error(f"Ошибка при освобождении блокировки: {e}")

# --- Загрузка переменных окружения --- #
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
BOT_USERNAME = os.getenv("TELEGRAM_BOT_USERNAME", "placeholder_bot_username").lstrip('@')
# Загружаем ID админов
ADMIN_USER_IDS_STR = os.getenv("ADMIN_USER_IDS", "")
ADMIN_IDS = set()
if ADMIN_USER_IDS_STR:
    try:
        ADMIN_IDS = {int(admin_id.strip()) for admin_id in ADMIN_USER_IDS_STR.split(",")}
        logger.info(f"Загружены ID администраторов: {ADMIN_IDS}")
    except ValueError:
        logger.error("Ошибка парсинга ADMIN_USER_IDS в .env. Убедитесь, что это числа через запятую.")

if not TELEGRAM_BOT_TOKEN:
    logger.critical("TELEGRAM_BOT_TOKEN не найден! Бот не может запуститься.")
    sys.exit(1)

# --- Инициализация OCR --- #
OCR_READER = None
try:
    OCR_READER = easyocr.Reader(['ru', 'en'], gpu=False)
    logger.info("OCR Reader (easyocr) инициализирован для языков [ru, en].")
except Exception as e:
    logger.error(f"Ошибка инициализации OCR Reader: {e}", exc_info=True)

# --- Инициализация Агента LangGraph --- #
try:
    agent_app = setup_agent()
except Exception as e:
    logger.critical(f"Ошибка при инициализации агента: {e}", exc_info=True)
    sys.exit(1)

# --- Вспомогательная функция для вызова агента ---
async def run_agent_for_user(user_id: int, chat_id: int, text: str, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"Вызов агента для user {user_id} в чате {chat_id}, текст: '{text}'")
    thread_id = f"telegram_{chat_id}" 
    config: Dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    inputs: Dict[str, Union[List[HumanMessage], int]] = {
        "messages": [HumanMessage(content=text)], 
        "user_id": user_id
    }
    final_response: str = "Извините, я не смог обработать ваш запрос."

    try:
        final_state = await agent_app.ainvoke(inputs, config=config)
        final_messages: list[AnyMessage] = final_state.get("messages", [])
        if final_messages and isinstance(final_messages[-1], AIMessage):
            final_response = final_messages[-1].content
            logger.info(f"Агент завершил работу для thread_id: {thread_id}.")
        else:
             logger.warning(f"Агент завершился, но не найдено финального AIMessage для thread_id: {thread_id}")
    except Exception as e:
        logger.error(f"Ошибка при вызове агента для user {user_id} в чате {chat_id}: {e}", exc_info=True)
        final_response = "Извините, произошла внутренняя ошибка при обработке вашего запроса."

    await context.bot.send_message(chat_id=chat_id, text=final_response)

# --- Обработчики Telegram --- #
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я QA-бот на базе YandexGPT. Спроси меня что-нибудь или пришли картинку с текстом!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    chat_id = message.chat_id
    user_id = message.from_user.id
    message_text = message.text

    if not message_text:
        return

    # Проверка, нужно ли боту отвечать
    should_respond = False
    if message.chat.type == "private":
        should_respond = True
    elif message.chat.type in ["group", "supergroup"]:
        mentioned = f"@{BOT_USERNAME}" in message_text
        is_reply_to_bot = (
            message.reply_to_message and 
            message.reply_to_message.from_user.username == BOT_USERNAME
        )
        if mentioned:
            should_respond = True
            message_text = message_text.replace(f"@{BOT_USERNAME}", "").strip()
            logger.info(f"Бот упомянут в чате {chat_id}.")
        elif is_reply_to_bot:
            should_respond = True
            logger.info(f"Сообщение является ответом на сообщение бота в чате {chat_id}.")

    if not should_respond:
        return

    await run_agent_for_user(user_id, chat_id, message_text, context)

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    chat_id = message.chat_id
    user_id = message.from_user.id

    if not OCR_READER:
        logger.warning(f"Получено фото от user {user_id} в чате {chat_id}, но OCR не инициализирован.")
        await message.reply_text("Извините, я пока не умею обрабатывать изображения.")
        return

    logger.info(f"Получено фото от user {user_id} в чате {chat_id}. Попытка распознать текст...")
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        photo_file = await message.photo[-1].get_file()
        file_bytes = await photo_file.download_as_bytearray()
        image_bytes = bytes(file_bytes)

        ocr_result = OCR_READER.readtext(image_bytes)

        if not ocr_result:
            logger.info("Текст на изображении не распознан.")
            await message.reply_text("Не удалось распознать текст на этом изображении.")
            return

        extracted_text = " ".join([res[1] for res in ocr_result])
        logger.info(f"Распознанный текст: {extracted_text[:200]}...")

        agent_input_text = f"Пользователь прислал картинку. Распознанный текст с картинки: '{extracted_text}'. Проанализируй этот текст или ответь на вопрос, если он есть в тексте."
        
        await run_agent_for_user(user_id, chat_id, agent_input_text, context)

    except Exception as e:
        logger.error(f"Ошибка при обработке фото от user {user_id} в чате {chat_id}: {e}", exc_info=True)
        await message.reply_text("Произошла ошибка при обработке изображения.")

async def cleanup_scheduler(context: ContextTypes.DEFAULT_TYPE):
    """Планировщик для очистки старых FAQ записей."""
    logger.info("Запуск планового удаления старых FAQ записей...")
    try:
        with connection.get_db_session() as db:
            if db:
                deleted = crud.cleanup_old_faq_entries(db)
                logger.info(f"Удалено {deleted} старых FAQ записей")
            else:
                logger.error("Не удалось получить сессию БД для очистки")
    except Exception as e:
        logger.error(f"Ошибка при выполнении очистки: {e}", exc_info=True)

async def admin_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /admin для управления админами."""
    message = update.message
    user_id = message.from_user.id
    username = message.from_user.username

    # Проверяем, является ли пользователь супер-админом (из .env)
    if str(user_id) not in ADMIN_USER_IDS_STR.split(","):
        await message.reply_text("У вас нет прав для использования этой команды.")
        return

    # Получаем аргументы команды
    args = context.args
    if not args:
        await message.reply_text("Использование: /admin add|remove|list user_id")
        return

    action = args[0].lower()
    
    try:
        with connection.get_db_session() as db:
            if not db:
                await message.reply_text("Ошибка подключения к БД")
                return

            if action == "list":
                admins = crud.get_all_active_admins(db)
                if not admins:
                    await message.reply_text("Список админов пуст")
                    return
                admin_list = "\n".join([f"ID: {admin.user_id}, Username: @{admin.username or 'N/A'}" for admin in admins])
                await message.reply_text(f"Список активных админов:\n{admin_list}")
                return

            if len(args) < 2:
                await message.reply_text("Необходимо указать user_id")
                return

            try:
                target_user_id = int(args[1])
            except ValueError:
                await message.reply_text("user_id должен быть числом")
                return

            if action == "add":
                if crud.add_admin(db, target_user_id):
                    await message.reply_text(f"Админ {target_user_id} успешно добавлен")
                else:
                    await message.reply_text("Ошибка при добавлении админа")

            elif action == "remove":
                if crud.deactivate_admin(db, target_user_id):
                    await message.reply_text(f"Админ {target_user_id} деактивирован")
                else:
                    await message.reply_text("Админ не найден или уже деактивирован")

            else:
                await message.reply_text("Неизвестное действие. Используйте: add, remove или list")

    except Exception as e:
        logger.error(f"Ошибка в команде admin: {e}", exc_info=True)
        await message.reply_text("Произошла ошибка при выполнении команды")

async def update_docs_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /update_docs для обновления документации."""
    message = update.message
    user_id = message.from_user.id

    # Проверяем, является ли пользователь админом
    if str(user_id) not in ADMIN_USER_IDS_STR.split(","):
        await message.reply_text("У вас нет прав для использования этой команды.")
        return

    try:
        await message.reply_text("Начинаю обновление документации Yandex Cloud...")
        
        with connection.get_db_session() as db:
            if not db:
                await message.reply_text("Ошибка подключения к БД")
                return
                
            from database.yandex_docs import YandexDocsManager
            docs_manager = YandexDocsManager()
            await docs_manager.update_docs(db)
            
        await message.reply_text("Документация успешно обновлена!")
        
    except Exception as e:
        logger.error(f"Ошибка при обновлении документации: {e}", exc_info=True)
        await message.reply_text(f"Произошла ошибка при обновлении документации: {e}")

async def handle_new_chat_members(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик добавления бота в группу."""
    chat_id = update.message.chat_id
    new_members = update.message.new_chat_members
    
    # Проверяем, добавлен ли наш бот
    bot_added = any(member.id == context.bot.id for member in new_members)
    
    if bot_added:
        logger.info(f"Бот добавлен в чат {chat_id}")
        try:
            # Загружаем историю чата
            loader = ChatHistoryLoaderTool()
            result = loader._run(chat_id=chat_id, days_to_load=30)
            logger.info(f"Результат загрузки истории: {result}")
            
            # Приветственное сообщение
            welcome_message = (
                "Привет! Я бот для поиска информации. "
                "Я уже загрузил историю чата и готов помочь с поиском. "
                "Чтобы задать вопрос, просто напишите его или упомяните меня через @."
            )
            await context.bot.send_message(chat_id=chat_id, text=welcome_message)
            
        except Exception as e:
            logger.error(f"Ошибка при обработке добавления бота в чат {chat_id}: {e}", exc_info=True)
            await context.bot.send_message(
                chat_id=chat_id,
                text="Произошла ошибка при инициализации бота. Пожалуйста, попробуйте позже."
            )

def handle_shutdown(signum, frame, lock_fd=None):
    """Обработчик сигналов завершения."""
    logger.info("Получен сигнал завершения, освобождаем ресурсы...")
    if lock_fd:
        release_lock(lock_fd)
    sys.exit(0)

def main():
    """Основная функция запуска бота."""
    lock_fd = acquire_lock()
    
    try:
        # Инициализация приложения
        application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        
        # Добавление обработчиков
        application.add_handler(CommandHandler("start", start))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
        application.add_handler(CommandHandler("admin", admin_command))
        application.add_handler(CommandHandler("update_docs", update_docs_command))
        application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, handle_new_chat_members))
        
        # Запуск бота в режиме polling
        logger.info("Бот запущен в режиме polling")
        application.run_polling(allowed_updates=Update.ALL_TYPES)
        
    except Exception as e:
        logger.critical(f"Критическая ошибка при запуске бота: {e}", exc_info=True)
    finally:
        release_lock(lock_fd)

if __name__ == "__main__":
    main()
