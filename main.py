import asyncio
import logging
import os
import re  # Импортируем regex для парсинга
import sys
from datetime import timedelta
import io # Для работы с байтами изображения
import easyocr # Для распознавания текста

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
from agent.agent_executor import setup_agent  # AgentState не нужен напрямую
from langgraph.graph.message import AnyMessage  # Для типизации
from langchain_core.messages import HumanMessage, AIMessage

# --- Инструменты больше не импортируем и не вызываем напрямую ---
# from agent.tools.add_faq import AddFAQTool, AddFAQInput
# from agent.tools.update_faq import UpdateFAQTool, UpdateFAQInput
# from agent.tools.delete_faq import DeleteFAQTool, DeleteFAQInput

# --- Конфигурация логирования --- #
LOG_DIR = "/app/logs"  # Путь внутри контейнера
LOG_FILE = os.path.join(LOG_DIR, "bot.log")

# Создаем директорию логов, если ее нет
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Вывод в консоль контейнера
        logging.FileHandler(LOG_FILE),  # Запись в файл
    ]
)
logger = logging.getLogger(__name__)

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
# Инициализируем один раз при старте (может занять время при первом запуске)
OCR_READER = None
try:
    # Указываем языки: русский и английский
    OCR_READER = easyocr.Reader(['ru', 'en'], gpu=False) # Используем CPU
    logger.info("OCR Reader (easyocr) инициализирован для языков [ru, en].")
except Exception as e:
    logger.error(f"Ошибка инициализации OCR Reader: {e}", exc_info=True)
    # Бот продолжит работать, но не сможет обрабатывать картинки

# --- Инициализация Агента LangGraph --- #
try:
    agent_app = setup_agent()
except Exception as e:
    logger.critical(f"Ошибка при инициализации агента: {e}", exc_info=True)
    sys.exit(1)

# --- Вспомогательная функция для вызова агента ---
async def run_agent_for_user(user_id: int, chat_id: int, text: str, context: ContextTypes.DEFAULT_TYPE):
    logger.info(f"Вызов агента для user {user_id} в чате {chat_id}, текст: '{text}'")
    thread_id = f"telegram_{chat_id}" 
    config = {"configurable": {"thread_id": thread_id}}
    # Передаем и сообщение, и user_id
    inputs = {"messages": [HumanMessage(content=text)], "user_id": user_id}
    final_response = "Извините, я не смог обработать ваш запрос." # Ответ по умолчанию

    try:
        # Используем ainvoke для простоты получения финального ответа
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

    # 1. Проверка на "кто тебя сделал?"
    # Проверка на 'кто тебя сделал?' теперь будет внутри LLM (системный промпт)
    # if cleaned_text == "кто тебя сделал?":
    #     logger.info(f"Ответ на '{cleaned_text}' в чате {chat_id}")
    #     await message.reply_text("Меня сделали в \"YandexGPT\"")
    #     return

    # 2. Проверка, нужно ли боту отвечать (в личке или если упомянули/ответили)
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
        return # Игнорируем сообщение

    # Вызываем общую функцию для запуска агента
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
        # Берем фото лучшего качества
        photo_file = await message.photo[-1].get_file()
        # Скачиваем в память
        file_bytes = await photo_file.download_as_bytearray()
        image_bytes = bytes(file_bytes)

        # Распознаем текст
        ocr_result = OCR_READER.readtext(image_bytes)

        if not ocr_result:
            logger.info("Текст на изображении не распознан.")
            await message.reply_text("Не удалось распознать текст на этом изображении.")
            return

        # Собираем распознанный текст
        extracted_text = " ".join([res[1] for res in ocr_result])
        logger.info(f"Распознанный текст: {extracted_text[:200]}...")

        # Формируем сообщение для агента
        agent_input_text = f"Пользователь прислал картинку. Распознанный текст с картинки: '{extracted_text}'. Проанализируй этот текст или ответь на вопрос, если он есть в тексте."
        
        # Вызываем агента с распознанным текстом
        await run_agent_for_user(user_id, chat_id, agent_input_text, context)

    except Exception as e:
        logger.error(f"Ошибка при обработке фото от user {user_id} в чате {chat_id}: {e}", exc_info=True)
        await message.reply_text("Произошла ошибка при обработке изображения.")

# --- Точка входа --- #
def main():
    logger.info("Запуск Telegram-бота...")
    logger.info(f"Имя пользователя бота для упоминаний: @{BOT_USERNAME}")
    if ADMIN_IDS:
        logger.info(f"Администраторы бота: {ADMIN_IDS}")
    else:
        logger.warning("Список администраторов пуст (ADMIN_USER_IDS не задан или некорректен в .env)")

    defaults = Defaults()
    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .defaults(defaults)
        .build()
    )

    # Добавляем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    # Добавляем обработчик для фото
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo)) 

    # Добавляем планировщик очистки в асинхронный цикл событий
    application.job_queue.run_custom(callback=cleanup_scheduler, job_kwargs={"name": "faq_cleanup"})

    logger.info("Бот запущен и готов принимать сообщения.")
    application.run_polling()

if __name__ == "__main__":
    main()
