import asyncio
import logging
from typing import Dict, Any, List, Union

from telegram import Update, BotCommand, Message
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from telegram.constants import ParseMode

from database import connection, models
from database.crud_admin import get_all_active_admins, add_admin, deactivate_admin
from database.embeddings import clear_embedding_cache
from .image_processor import process_image_and_run_agent
from .agent_utils import run_agent_for_user

logger = logging.getLogger(__name__)


# --- Обработчики команд --- #
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отправляет приветственное сообщение при команде /start."""
    user_name = update.effective_user.first_name
    await update.message.reply_text(
        f"Привет, {user_name}! Я QA-бот команды поддержки Yandex Cloud. Чем могу помочь?\n"
        f"Задайте мне вопрос о сервисах Yandex Cloud или нашей внутренней базе знаний."
    )


async def admin_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /admin для управления админами."""
    message = update.message
    user_id = message.from_user.id
    ADMIN_IDS = context.application.bot_data.get("admin_ids", set())
    if user_id not in ADMIN_IDS:
        await message.reply_text("У вас нет прав для использования этой команды.")
        return
    args = context.args
    if not args or len(args) < 1:
        await message.reply_text(
            "Использование: /admin list | add <user_id> [username] | remove <user_id>"
        )
        return
    action = args[0].lower()
    try:
        with connection.get_db_session() as db:
            if not db:
                await message.reply_text(
                    "Ошибка: Не удалось получить сессию базы данных."
                )
                return
            if action == "list":
                admins = get_all_active_admins(db)
                if not admins:
                    await message.reply_text("Список админов пуст.")
                else:
                    admin_list = "\n".join(
                        [
                            f"ID: `{admin.user_id}`, Username: @{admin.username or 'N/A'}, Added: {admin.created_at.strftime('%Y-%m-%d')}"
                            for admin in admins
                        ]
                    )
                    await message.reply_text(
                        f"*Список активных админов:*\n{admin_list}",
                        parse_mode=ParseMode.MARKDOWN,
                    )
                return
            if action in ["add", "remove"]:
                if len(args) < 2:
                    await message.reply_text(
                        f"Необходимо указать user_id для действия '{action}'."
                    )
                    return
                try:
                    target_user_id = int(args[1])
                except ValueError:
                    await message.reply_text("user_id должен быть числом.")
                    return
                if action == "add":
                    target_username = args[2] if len(args) > 2 else None
                    if target_username and target_username.startswith("@"):
                        target_username = target_username[1:]
                    added_admin = add_admin(db, target_user_id, target_username)
                    if added_admin:
                        await message.reply_text(
                            f"Админ ID:`{target_user_id}` (Username: @{added_admin.username or 'N/A'}) успешно добавлен.",
                            parse_mode=ParseMode.MARKDOWN,
                        )
                    else:
                        await message.reply_text(
                            "Ошибка при добавлении админа (возможно, ошибка БД).",
                            parse_mode=ParseMode.MARKDOWN,
                        )
                elif action == "remove":
                    if deactivate_admin(db, target_user_id):
                        await message.reply_text(
                            f"Админ ID:`{target_user_id}` деактивирован.",
                            parse_mode=ParseMode.MARKDOWN,
                        )
                    else:
                        await message.reply_text(
                            "Админ не найден или уже деактивирован.",
                            parse_mode=ParseMode.MARKDOWN,
                        )
            else:
                await message.reply_text(
                    "Неизвестное действие. Используйте: list, add, remove"
                )
    except Exception as e:
        logger.error(f"Ошибка в команде admin: {e}", exc_info=True)
        await message.reply_text("Произошла ошибка при выполнении команды.")


async def clear_cache_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /clearcache."""
    user_id = update.effective_user.id
    ADMIN_IDS = context.application.bot_data.get("admin_ids", set())
    if user_id not in ADMIN_IDS:
        await update.message.reply_text("У вас нет прав для очистки кеша.")
        return
    try:
        cleared_embeddings = clear_embedding_cache()
        redis_cache = context.application.bot_data.get("redis_cache")
        if redis_cache:
            cleared_redis = redis_cache.clear_all_caches()
            await update.message.reply_text(
                f"Кеш Redis очищен ({cleared_redis} ключей удалено). Кеш эмбеддингов очищен ({cleared_embeddings} записей удалено)."
            )
        else:
            await update.message.reply_text(
                f"Кеш Redis не настроен. Кеш эмбеддингов очищен ({cleared_embeddings} записей удалено)."
            )
    except Exception as e:
        logger.error(f"Ошибка при очистке кеша: {e}", exc_info=True)
        await update.message.reply_text(f"Произошла ошибка при очистке кеша: {e}")


# --- Обработчики сообщений --- #
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает текстовые сообщения от пользователя."""
    if not update.message or not update.message.text or not update.message.from_user:
        logger.debug("Пустое или некорректное сообщение получено, игнорируется.")
        return

    user_id = update.message.from_user.id
    chat_id = update.message.chat_id
    text = update.message.text
    bot_username = context.application.bot_data.get("bot_username")
    actual_bot_username = context.bot.username if context.bot.username else bot_username

    mentioned_or_reply_or_private = False
    if update.message.chat.type == "private":
        mentioned_or_reply_or_private = True
    elif update.message.chat.type in ["group", "supergroup"]:
        if actual_bot_username and f"@{actual_bot_username}" in text:
            mentioned_or_reply_or_private = True
            text = text.replace(f"@{actual_bot_username}", "").strip()
            if not text:
                logger.info(
                    f"Сообщение в группе {chat_id} содержало только упоминание бота. Отвечаем приветствием."
                )
                await update.message.reply_text(
                    "Да, я здесь! Чем могу помочь?",
                    reply_to_message_id=update.message.message_id,
                )
                return
        elif (
            update.message.reply_to_message
            and update.message.reply_to_message.from_user
            and update.message.reply_to_message.from_user.username
            == actual_bot_username
        ):
            mentioned_or_reply_or_private = True

    if not mentioned_or_reply_or_private:
        logger.debug(
            f"Сообщение в группе {chat_id} без упоминания/ответа боту или не в ЛС, игнорируется."
        )
        return

    if text.strip().lower() in [
        "спасибо",
        "спасибо!",
        "благодарю",
        "спс",
        "thx",
        "thanks",
    ]:
        await update.message.reply_text(
            "Пожалуйста! Рад был помочь. Если будут ещё вопросы — обращайтесь."
        )
        return

    if "это не то" in text.lower() or "не совсем то" in text.lower():
        await update.message.reply_text(
            "Понял, извините. Пожалуйста, уточните ваш вопрос или что именно вы хотели бы узнать, чтобы я мог дать более точный ответ."
        )
        return

    logger.info(
        f"Вызов агента для user {user_id} в чате {chat_id} (упомянут/ЛС: {mentioned_or_reply_or_private}), текст: '{text[:50]}...'"
    )
    asyncio.create_task(run_agent_for_user(user_id, chat_id, text, context))


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает полученные изображения (фотографии)."""
    if not update.message or not update.message.photo or not update.message.from_user:
        logger.debug(
            "Сообщение не содержит фото или информации о пользователе, игнорируется."
        )
        return

    user_id = update.message.from_user.id
    chat_id = update.message.chat_id
    bot_username = context.application.bot_data.get("bot_username")
    actual_bot_username = context.bot.username if context.bot.username else bot_username
    photo_file_id = update.message.photo[-1].file_id
    text_from_caption = update.message.caption or ""

    process_ocr = False
    if update.message.chat.type == "private":
        process_ocr = True
    elif update.message.chat.type in ["group", "supergroup"]:
        if actual_bot_username and f"@{actual_bot_username}" in text_from_caption:
            process_ocr = True
            text_from_caption = text_from_caption.replace(
                f"@{actual_bot_username}", ""
            ).strip()
        elif (
            update.message.reply_to_message
            and update.message.reply_to_message.from_user
            and update.message.reply_to_message.from_user.username
            == actual_bot_username
        ):
            process_ocr = True

    if not process_ocr:
        logger.info(
            f"Фото в чате {chat_id} (не ЛС) без явного указания на обработку OCR (упоминание/ответ боту в подписи), игнорируется."
        )
        return

    processing_msg = await update.message.reply_text(
        "Получил фото, начинаю распознавание текста...",
        reply_to_message_id=update.message.message_id,
    )

    asyncio.create_task(
        process_image_and_run_agent(
            photo_file_id, user_id, chat_id, text_from_caption, context, processing_msg
        )
    )


async def handle_new_chat_members(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отправляет приветственное сообщение новым участникам чата, если среди них бот."""
    message = update.message
    if not message or not message.new_chat_members:
        return

    chat_id = message.chat_id
    new_members = message.new_chat_members
    bot_username = context.application.bot_data.get("bot_username")
    bot_added = any(member.username == bot_username for member in new_members)

    if bot_added:
        logger.info(f"Бот добавлен в чат {chat_id} ({message.chat.title})")
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"Привет! Я QA-бот Yandex Cloud. Задавайте мне вопросы о сервисах Yandex Cloud или нашей базе знаний, упомянув меня (@{{bot_username}}) или ответив на мое сообщение.",
        )


# Новая функция для регистрации всех обработчиков
def setup_handlers(application: Application):
    """Регистрирует все обработчики команд и сообщений."""
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("admin", admin_command))
    application.add_handler(CommandHandler("clearcache", clear_cache_command))

    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(
        MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, handle_new_chat_members)
    )


# --- Команды для установки через BotFather --- #
DEFAULT_COMMANDS = [
    BotCommand("start", "Запустить бота и получить приветствие"),
    BotCommand("admin", "[Админ] Управление админами (list/add/remove)"),
    BotCommand("clearcache", "[Админ] Очистить кеш Redis и эмбеддингов"),
]
