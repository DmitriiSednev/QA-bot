import asyncio
import logging
import os
import re  # Импортируем regex для парсинга

from dotenv import load_dotenv
from telegram.ext import Application, CommandHandler, MessageHandler, filters

# --- Import Agent Logic ---
from agent import setup_agent  # AgentState не нужен напрямую
from langchain_core.messages import HumanMessage

# --- Инструменты больше не импортируем и не вызываем напрямую ---
# from agent.tools.add_faq import AddFAQTool, AddFAQInput
# from agent.tools.update_faq import UpdateFAQTool, UpdateFAQInput
# from agent.tools.delete_faq import DeleteFAQTool, DeleteFAQInput

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)  # Quieten httpx logs
logger = logging.getLogger(__name__)

# --- Placeholder Handlers (will be replaced by agent logic) ---


async def start(update, context):
    """Sends a welcome message when the /start command is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Привет {user.mention_html()}! Я QA бот. Спроси меня что-нибудь.",
    )


async def handle_message(update, context):
    """Handles regular messages, passing them to the agent executor."""
    user_message_text = update.message.text
    chat_id = update.message.chat_id  # Used as thread_id

    logger.info(f"Получено сообщение от {chat_id}: {user_message_text}")

    if "agent_executor" not in context.bot_data:
        logger.error("Agent executor не инициализирован в context.bot_data!")
        await update.message.reply_text("Ошибка: Агент не готов.")
        return

    agent_executor = context.bot_data["agent_executor"]
    config = {"configurable": {"thread_id": str(chat_id)}}

    # Просто передаем сообщение пользователя агенту
    initial_state = {"messages": [HumanMessage(content=user_message_text)]}

    # --- Вызов агента --- (всегда с одним сообщением)
    try:
        logger.info(f"Вызов агента с начальным состоянием: {initial_state}")
        final_state = await agent_executor.ainvoke(initial_state, config=config)
        logger.info(f"Полное состояние от агента: {final_state}")

        # --- Извлечение ответа --- (ищем последнее сообщение AI)
        agent_response = "(Пустой ответ от агента)"
        if final_state and "messages" in final_state and final_state["messages"]:
            last_message = final_state["messages"][-1]

            # Ищем последний ответ AI
            if last_message.type == "ai":
                agent_response = last_message.content
            else:
                # Если последний - не AI (например, ToolMessage), ищем предыдущий AI
                logger.warning(
                    f"Последнее сообщение не AI ({last_message.type}). Ищем предыдущее AI."
                )
                for msg in reversed(final_state["messages"]):
                    if msg.type == "ai":
                        agent_response = msg.content
                        break

        await update.message.reply_text(agent_response)

    except Exception as e:
        logger.error(f"Ошибка при вызове агента для чата {chat_id}: {e}", exc_info=True)
        await update.message.reply_text(
            "Произошла ошибка при обработке вашего запроса."
        )


# --- Command Handlers --- (УДАЛЯЕМ add_faq_command, update_faq_command, delete_faq_command)

# async def add_faq_command(update, context): ... (удалено)
# async def update_faq_command(update, context): ... (удалено)
# async def delete_faq_command(update, context): ... (удалено)


# --- Main Application Setup ---


def main() -> None:
    """Start the bot."""
    # Load environment variables
    load_dotenv()

    # Get Telegram Bot Token
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN не найден в .env файле!")
        return

    # Create the Application
    application = Application.builder().token(token).build()

    # --- Setup Agent --- (Вызываем один раз при старте)
    try:
        agent_executor = setup_agent()
        # Store the agent executor in bot_data for access in handlers
        application.bot_data["agent_executor"] = agent_executor
        logger.info("Agent executor успешно создан и сохранен в bot_data.")
    except Exception as e:
        logger.error(
            f"Критическая ошибка при создании agent_executor: {e}", exc_info=True
        )
        # Exit or handle gracefully if the agent is essential
        return

    # --- Register Handlers ---
    # Register the /start command handler
    application.add_handler(CommandHandler("start", start))

    # Register command handlers (УДАЛЯЕМ add/update/delete)
    # application.add_handler(CommandHandler("add_faq", add_faq_command))
    # application.add_handler(CommandHandler("update_faq", update_faq_command))
    # application.add_handler(CommandHandler("delete_faq", delete_faq_command))

    # Register the message handler for non-command messages
    # Используем filters.TEXT и НЕ filters.COMMAND, чтобы он ловил всё, КРОМЕ команд ТГ
    # Наш парсинг внутри handle_message разберется с "командами" типа /add_faq
    application.add_handler(MessageHandler(filters.TEXT, handle_message))

    # Run the bot until the user presses Ctrl-C
    logger.info("Бот запускается...")
    application.run_polling()


if __name__ == "__main__":
    main()
