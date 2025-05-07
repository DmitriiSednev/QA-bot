import asyncio
import logging
import os
import io
from typing import Dict, Any, List, Union

import easyocr
from telegram import Update, BotCommand
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

from database import connection, models
from database.crud_admin import get_all_active_admins, add_admin, deactivate_admin
from database.embeddings import clear_embedding_cache
from database.crud_faq import search_faq_entries, add_faq_entry

# Импортируем скомпилированный агент
# Имя переменной agent_app должно совпадать с тем, что используется в main.py
# Если main.py импортирует setup_agent, то нам нужно передать agent_app сюда
# Или сделать agent_app глобальным/синглтоном, что не очень хорошо.
# Проще всего импортировать setup_agent и вызывать его здесь, но это может
# привести к повторной инициализации, если main.py тоже его вызывает.
# Вариант: передавать agent_app через context.bot_data или context.application.bot_data

# Пока оставим зависимость от main.py для agent_app и OCR_READER
# from main import agent_app, OCR_READER, ADMIN_IDS, BOT_USERNAME
# Правильнее будет передавать зависимости через context.application.bot_data

from langgraph.graph.message import AnyMessage
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)


# --- Вспомогательная функция для вызова агента --- #
async def run_agent_for_user(
    user_id: int, chat_id: int, text: str, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Вызывает LangGraph агент для обработки сообщения пользователя."""
    agent_app = context.application.bot_data.get("agent_app")
    if not agent_app:
        logger.error("Агент (agent_app) не найден в bot_data!")
        await context.bot.send_message(
            chat_id=chat_id, text="Ошибка конфигурации бота."
        )
        return

    logger.info(f"Вызов агента для user {user_id} в чате {chat_id}, текст: '{text}'")
    thread_id = f"telegram_{chat_id}_{user_id}"  # Более уникальный ID потока
    config: Dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    # Передаем user_id в состояние агента
    inputs: Dict[str, Union[List[HumanMessage], int]] = {
        "messages": [HumanMessage(content=text)],
        "user_id": user_id,
    }
    final_response: str = "Извините, я не смог обработать ваш запрос."

    try:
        # Используем astream_events для более детального контроля и логирования
        final_state = None
        async for event in agent_app.astream_events(
            inputs, config=config, version="v1"
        ):
            kind = event["event"]
            name = event.get("name", "")
            # logger.debug(f"Agent Event: {kind} - {name} - Data: {event.get('data')}")
            if (
                kind == "on_chain_end" and name == "LangGraph"
            ):  # Отслеживаем конец всего графа
                final_state = event["data"]["output"]
                logger.debug(f"Agent finished. Final state keys: {final_state.keys()}")
                break  # Выходим из цикла после получения финального состояния

        # final_state = await agent_app.ainvoke(inputs, config=config)

        if final_state:
            final_messages: list[AnyMessage] = final_state.get("messages", [])
            if final_messages and isinstance(final_messages[-1], AIMessage):
                final_response = final_messages[-1].content
                logger.info(
                    f"Агент завершил работу для thread_id: {thread_id}. Ответ: {final_response[:100]}..."
                )

                # --- Логика добавления в FAQ (можно вынести) ---
                # Добавляем в FAQ, если в запросе было что-то про Yandex и ответ не был найден в FAQ
                # Это временное решение, лучше иметь явный шаг/инструмент для этого
                if "яндекс" in text.lower():
                    try:
                        with connection.get_db_session() as db:
                            if db:
                                # Проверяем, есть ли уже ОЧЕНЬ похожий вопрос
                                similar = search_faq_entries(
                                    db, text, limit=1, min_similarity=0.95
                                )  # Более строгий порог
                                if not similar:
                                    logger.info(
                                        f"Добавляю новый вопрос про Яндекс в FAQ: {text[:50]}..."
                                    )
                                    add_faq_entry(
                                        db, question=text, answer=final_response
                                    )
                                else:
                                    logger.info(
                                        "Похожий вопрос про Яндекс уже есть в FAQ, не добавляю."
                                    )
                            else:
                                logger.error(
                                    "Не удалось получить сессию БД для добавления в FAQ."
                                )
                    except Exception as db_err:
                        logger.error(
                            f"Ошибка при попытке добавить запись в FAQ: {db_err}",
                            exc_info=True,
                        )
                # --- Конец логики добавления в FAQ ---

            else:
                logger.warning(
                    f"Агент завершился, но не найдено финального AIMessage для thread_id: {thread_id}. State: {final_state}"
                )
                # Попытка извлечь последний AIMessage, если он не последний в списке
                ai_messages = [m for m in final_messages if isinstance(m, AIMessage)]
                if ai_messages:
                    final_response = ai_messages[-1].content
                    logger.info(
                        f"Извлечен последний AIMessage: {final_response[:100]}..."
                    )
                else:
                    final_response = (
                        "Не удалось получить финальный ответ от ассистента."
                    )
        else:
            logger.error(
                f"Агент не вернул финальное состояние для thread_id: {thread_id}"
            )
            final_response = "Ошибка: Агент не вернул результат."

    except Exception as e:
        logger.error(
            f"Ошибка при вызове агента для user {user_id} в чате {chat_id}: {e}",
            exc_info=True,
        )
        final_response = (
            "Извините, произошла внутренняя ошибка при обработке вашего запроса."
        )

    await context.bot.send_message(
        chat_id=chat_id, text=final_response, parse_mode=ParseMode.MARKDOWN
    )


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
    username = message.from_user.username
    ADMIN_IDS = context.application.bot_data.get("admin_ids", set())

    # Проверяем, является ли пользователь супер-админом (из .env)
    if user_id not in ADMIN_IDS:
        await message.reply_text("У вас нет прав для использования этой команды.")
        return

    # Получаем аргументы команды
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
                    # Опционально принимаем username
                    target_username = args[2] if len(args) > 2 else None
                    # Удаляем @ если есть
                    if target_username and target_username.startswith("@"):
                        target_username = target_username[1:]

                    added_admin = add_admin(db, target_user_id, target_username)
                    if added_admin:
                        await message.reply_text(
                            f"Админ ID:`{target_user_id}` (Username: @{added_admin.username or 'N/A'}) успешно добавлен.",
                            parse_mode=ParseMode.MARKDOWN,
                        )
                    else:
                        # add_admin возвращает существующего или нового админа, None только при ошибке БД
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
        # Очистка кеша эмбеддингов (если есть)
        cleared_embeddings = clear_embedding_cache()

        # Очистка кеша Redis (если есть)
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
    """Обрабатывает текстовые сообщения."""
    message = update.message
    if (
        not message or not message.text
    ):  # Проверка на пустые сообщения или обновления без текста
        return

    chat_id = message.chat_id
    user_id = message.from_user.id
    message_text = message.text
    BOT_USERNAME = context.application.bot_data.get("bot_username")

    # Реакция на "спасибо"
    if message_text.strip().lower() in [
        "спасибо",
        "спасибо!",
        "благодарю",
        "спс",
        "thx",
        "thanks",
    ]:
        await message.reply_text(
            "Пожалуйста! Рад был помочь. Если будут ещё вопросы — обращайтесь."
        )
        return

    # Реакция на "это не то"
    if "это не то" in message_text.lower() or "не совсем то" in message_text.lower():
        await message.reply_text(
            "Понял, извините. Пожалуйста, уточните ваш вопрос или что именно вы хотели бы узнать, чтобы я мог дать более точный ответ."
        )
        return

    should_respond = False
    # Отвечаем всегда в личных сообщениях
    if message.chat.type == "private":
        should_respond = True
    # В группах отвечаем только при упоминании или ответе на сообщение бота
    elif message.chat.type in ["group", "supergroup"]:
        mentioned = BOT_USERNAME and f"@{BOT_USERNAME}" in message_text
        is_reply_to_bot = (
            message.reply_to_message
            and message.reply_to_message.from_user.username == BOT_USERNAME
        )

        if mentioned:
            should_respond = True
            # Удаляем упоминание бота из текста
            message_text = message_text.replace(f"@{BOT_USERNAME}", "").strip()
            logger.info(
                f"Бот упомянут в группе {chat_id}. Текст: '{message_text[:50]}...' "
            )
        elif is_reply_to_bot:
            should_respond = True
            logger.info(
                f"Ответ на сообщение бота в группе {chat_id}. Текст: '{message_text[:50]}...' "
            )

    if not should_respond:
        # logger.debug(f"Сообщение в чате {chat_id} проигнорировано (не личное, не упоминание, не ответ).")
        return

    # Проверка на пустое сообщение после удаления упоминания
    if not message_text.strip():
        await message.reply_text("Пожалуйста, задайте ваш вопрос после упоминания.")
        return

    # Запускаем агент
    await run_agent_for_user(user_id, chat_id, message_text, context)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обрабатывает сообщения с фотографиями (OCR)."""
    message = update.message
    if not message or not message.photo:
        return

    chat_id = message.chat_id
    user_id = message.from_user.id
    OCR_READER = context.application.bot_data.get("ocr_reader")

    if not OCR_READER:
        logger.warning(
            f"Получено фото от user {user_id} в чате {chat_id}, но OCR не инициализирован."
        )
        # Отвечаем только если бота упомянули с картинкой или в личке
        if message.chat.type == "private" or (
            message.caption
            and context.application.bot_data.get("bot_username") in message.caption
        ):
            await message.reply_text(
                "Извините, я пока не умею обрабатывать изображения."
            )
        return

    logger.info(
        f"Получено фото от user {user_id} в чате {chat_id}. Попытка распознать текст..."
    )
    # Показываем "печатает..."
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        photo_file = await message.photo[
            -1
        ].get_file()  # Берем самое большое разрешение
        # Скачиваем фото в память
        file_bytes_io = io.BytesIO()
        await photo_file.download_to_memory(file_bytes_io)
        image_bytes = file_bytes_io.getvalue()

        # Запускаем OCR в executor'е, чтобы не блокировать основной поток
        loop = asyncio.get_running_loop()
        # Используем partial, если нужно передать аргументы
        # from functools import partial
        # ocr_task = partial(OCR_READER.readtext, image_bytes, detail=0) # detail=0 возвращает только текст
        # extracted_text_list = await loop.run_in_executor(None, ocr_task)

        # readtext ожидает путь к файлу или байты
        ocr_result = await loop.run_in_executor(None, OCR_READER.readtext, image_bytes)

        if not ocr_result:
            logger.info("Текст на изображении не распознан.")
            await message.reply_text("Не удалось распознать текст на этом изображении.")
            return

        # Собираем распознанный текст
        extracted_text = " ".join([res[1] for res in ocr_result])
        logger.info(
            f"Распознанный текст ({len(extracted_text)} chars): {extracted_text[:200]}..."
        )

        # Формируем запрос для агента
        # Если есть подпись к картинке, добавляем ее
        caption = message.caption or ""
        if caption:
            # Удаляем упоминание бота, если есть
            bot_username = context.application.bot_data.get("bot_username")
            if bot_username:
                caption = caption.replace(f"@{bot_username}", "").strip()
            agent_input_text = f"Пользователь прислал картинку с подписью '{caption}'. Распознанный текст с картинки: '{extracted_text}'. Ответь на вопрос из подписи, используя текст с картинки как контекст."
        else:
            agent_input_text = f"Пользователь прислал картинку. Распознанный текст с картинки: '{extracted_text}'. Проанализируй этот текст или ответь на вопрос, если он есть в тексте."

        # Запускаем агент
        await run_agent_for_user(user_id, chat_id, agent_input_text, context)

    except Exception as e:
        logger.error(
            f"Ошибка при обработке фото от user {user_id} в чате {chat_id}: {e}",
            exc_info=True,
        )
        await message.reply_text("Произошла ошибка при обработке изображения.")


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
            text="Привет! Я QA-бот Yandex Cloud. Задавайте мне вопросы о сервисах Yandex Cloud или нашей базе знаний, упомянув меня (@{bot_username}) или ответив на мое сообщение.",
        )


# Новая функция для регистрации всех обработчиков
def setup_handlers(application: Application):
    """Регистрирует все обработчики команд и сообщений."""
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("admin", admin_command))
    application.add_handler(CommandHandler("clearcache", clear_cache_command))
    # Добавь здесь другие CommandHandler, если они есть

    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(
        MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, handle_new_chat_members)
    )
    # Добавь здесь другие MessageHandler, CallbackQueryHandler и т.д.
    # Пример:
    # from telegram.ext import CallbackQueryHandler
    # application.add_handler(CallbackQueryHandler(button_callback))


# --- Команды для установки через BotFather --- #
DEFAULT_COMMANDS = [
    BotCommand("start", "Запустить бота и получить приветствие"),
    BotCommand("help", "Получить помощь (пока не реализовано)"),
    BotCommand("admin", "[Админ] Управление админами (list/add/remove)"),
    BotCommand("clearcache", "[Админ] Очистить кеш Redis и эмбеддингов"),
]
