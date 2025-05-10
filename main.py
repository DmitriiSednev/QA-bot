import asyncio
import logging
import os
import re  # Импортируем regex для парсинга

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# --- Import Agent Logic ---
from agent.agent_executor import setup_agent
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# --- Инструменты больше не импортируем и не вызываем напрямую ---
# from agent.tools.add_faq import AddFAQTool, AddFAQInput
# from agent.tools.update_faq import UpdateFAQTool, UpdateFAQInput
# from agent.tools.delete_faq import DeleteFAQTool, DeleteFAQInput

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Устанавливаем INFO, чтобы видеть сообщения от агента
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Вывод в консоль
        # logging.FileHandler("bot.log") # Опционально: запись в файл
    ],
)
logging.getLogger("httpx").setLevel(logging.WARNING)  # Quieten httpx logs
logger = logging.getLogger(__name__)

# --- Placeholder Handlers (will be replaced by agent logic) ---

# Глобальная переменная для хранения экземпляра агента (скомпилированного графа)
# Это упрощение для примера. В продакшене рассмотрите более надежное управление состоянием.
global_agent_executor = None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start."""
    logger.info(f"Команда /start от пользователя {update.effective_user.id}")
    await update.message.reply_text(
        "Привет! Я QA-бот на основе LangGraph. Задайте мне вопрос."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает текстовые сообщения от пользователя."""
    global global_agent_executor
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    message_text = update.message.text

    logger.info(
        f"Сообщение от пользователя {user_id} в чате {chat_id}: '{message_text}'"
    )

    if global_agent_executor is None:
        logger.error("Агент не инициализирован. Сообщение не будет обработано.")
        await update.message.reply_text(
            "Извините, у меня технические неполадки. Агент не загружен."
        )
        return

    # Уникальный ID для сессии пользователя (можно использовать user_id или chat_id)
    # Для LangGraph важно передавать один и тот же thread_id для продолжения диалога
    thread_id = f"telegram_{user_id}_{chat_id}"
    config = {"configurable": {"thread_id": thread_id}}

    # Начальное состояние для агента (только сообщение пользователя)
    # AgentState ожидает messages как Sequence[BaseMessage]
    initial_state = {"messages": [HumanMessage(content=message_text)]}

    try:
        # Используем stream для получения событий от графа
        # Вместо invoke, чтобы иметь возможность обрабатывать промежуточные шаги или просто получить конечный результат
        final_response_content = ""
        async for event in global_agent_executor.astream(initial_state, config=config):
            # logger.debug(f"Событие от агента: {event}") # Очень многословно
            # Нас интересует конечное событие, которое обычно является словарем с ключом "messages"
            # или конкретный узел, возвращающий финальный AIMessage.
            # Ищем ключ 'messages' в данных события последнего узла (END)
            # Структура событий зависит от версии LangGraph и того, как настроен граф.
            # Обычно последнее событие от узла перед END содержит финальное состояние.

            # Пример: если последний узел, который мы ожидаем перед END, это 'output_guardrails'
            # и он возвращает состояние с обновленными messages.
            # Или если мы смотрим на событие от самого узла END, у него есть data['output']
            if (
                event.get("event") == "on_chain_end"
                and event.get("name") == "LangGraph"
            ):  # Общее завершение графа
                final_state = event.get("data", {}).get("output")
                if (
                    final_state
                    and isinstance(final_state, dict)
                    and "messages" in final_state
                ):
                    if final_state["messages"] and isinstance(
                        final_state["messages"][-1], AIMessage
                    ):
                        final_response_content = final_state["messages"][-1].content
                        logger.info(
                            f"Финальный ответ агента для пользователя {user_id}: {final_response_content}"
                        )
                        break  # Получили финальный ответ

        if not final_response_content:
            # Если после стрима не нашли подходящего ответа (маловероятно, если граф доходит до END)
            logger.warning(
                f"Не удалось извлечь финальный ответ из стрима агента для пользователя {user_id}."
            )
            # Попытка получить последнее сообщение из invoke, если stream не дал результата
            # Это может быть медленнее, так как invoke ждет полного выполнения
            # final_state_invoke = await global_agent_executor.ainvoke(initial_state, config=config)
            # if final_state_invoke and "messages" in final_state_invoke and final_state_invoke["messages"]:
            #     final_response_content = final_state_invoke["messages"][-1].content
            # else:
            #     final_response_content = "Не удалось получить ответ от агента."
            # logger.info(f"Финальный ответ агента (через invoke fallback) для пользователя {user_id}: {final_response_content}")
            # Пока что оставим сообщение об ошибке, если стрим не дал ответа
            final_response_content = (
                "К сожалению, я не смог обработать ваш запрос полностью."
            )

    except Exception as e:
        logger.error(
            f"Ошибка при вызове агента для пользователя {user_id}: {e}", exc_info=True
        )
        final_response_content = (
            f"Извините, произошла ошибка при обработке вашего запроса: {e}"
        )

    await update.message.reply_text(final_response_content)


# --- Command Handlers --- (УДАЛЯЕМ add_faq_command, update_faq_command, delete_faq_command)

# async def add_faq_command(update, context): ... (удалено)
# async def update_faq_command(update, context): ... (удалено)
# async def delete_faq_command(update, context): ... (удалено)


# --- Main Application Setup ---


def main() -> None:
    """Запускает бота."""
    global global_agent_executor
    load_dotenv()  # Загружаем переменные окружения из .env файла
    logger.info("Загружены переменные окружения.")

    # Инициализация агента
    try:
        logger.info("Инициализация агента...")
        global_agent_executor = setup_agent()
        if global_agent_executor:
            logger.info("Агент успешно инициализирован.")
        else:
            logger.error(
                "Не удалось инициализировать агент! Бот может работать некорректно."
            )
            # Можно здесь завершить работу, если агент критичен
            # return
    except Exception as e:
        logger.error(f"Критическая ошибка при инициализации агента: {e}", exc_info=True)
        # Завершаем работу, если агент не может быть создан
        return

    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not telegram_bot_token:
        logger.error(
            "TELEGRAM_BOT_TOKEN не найден в переменных окружения. Бот не может быть запущен."
        )
        return

    logger.info("Создание приложения Telegram...")
    application = Application.builder().token(telegram_bot_token).build()

    # Регистрация обработчиков
    application.add_handler(CommandHandler("start", start))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )
    logger.info("Обработчики команд и сообщений зарегистрированы.")

    # Запуск бота
    logger.info("Запуск бота...")
    application.run_polling()
    logger.info("Бот остановлен.")


if __name__ == "__main__":
    main()
