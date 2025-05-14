import asyncio
import logging
from typing import Dict, Any, List, Union

from telegram.ext import ContextTypes
from telegram import (
    Message,
)  # Убедимся, что Message импортирован, если он нужен в run_agent_for_user (похоже, нет)
from telegram.constants import ParseMode

from langgraph.graph.message import AnyMessage
from langchain_core.messages import HumanMessage, AIMessage

# Импорты, которые могут понадобиться для run_agent_for_user, если он делает что-то с БД напрямую
# (в текущей версии он не делает, но для полноты картины)
from database import connection
from database.crud_faq import search_faq_entries, add_faq_entry


logger = logging.getLogger(__name__)


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
    thread_id = f"telegram_{chat_id}_{user_id}"
    config: Dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    inputs: Dict[str, Union[List[HumanMessage], int]] = {
        "messages": [HumanMessage(content=text)],
        "user_id": user_id,  # Передаем user_id в состояние агента
    }
    final_response: str = "Извините, я не смог обработать ваш запрос."

    try:
        # Показываем "печатает..."
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")

        # Устанавливаем таймаут для всего процесса
        async with asyncio.timeout(60):  # 60 секунд на весь процесс
            final_state_output = None
            full_final_state = (
                None  # Для отладки, если понадобится посмотреть все состояние
            )

            logger.debug(f"Запуск agent_app.astream_events для thread_id: {thread_id}")
            async for event in agent_app.astream_events(
                inputs,
                config=config,
                version="v1",  # или v2, если используете streaming v2
            ):
                kind = event["event"]
                name = event.get("name", "")
                tags = event.get("tags", [])

                # Логируем важные события
                if kind in [
                    "on_chain_start",
                    "on_chain_end",
                    "on_chat_model_stream",  # Если используется стриминг от LLM
                    "on_tool_start",
                    "on_tool_end",
                ]:
                    logger.debug(f"Agent Event: {kind} | Name: {name} | Tags: {tags}")

                if (
                    kind == "on_chain_end" and name == "LangGraph"
                ):  # Имя графа по умолчанию
                    full_final_state = event["data"].get("output")
                    if isinstance(full_final_state, dict):
                        final_state_output = full_final_state
                        logger.info(
                            f"Agent finished. Получен финальный словарь состояния: {list(final_state_output.keys())}"
                        )
                    else:  # Если граф вернул не словарь (например, только список сообщений)
                        logger.warning(
                            f'Agent finished, но event["data"]["output"] не является словарем. Тип: {type(full_final_state)}'
                        )
                        # Попытка извлечь сообщения, если это был прямой вывод списка сообщений
                        if isinstance(full_final_state, list) and all(
                            isinstance(m, (HumanMessage, AIMessage))
                            for m in full_final_state
                        ):
                            final_state_output = {"messages": full_final_state}
                            logger.info(
                                "Предполагаем, что граф вернул список сообщений напрямую."
                            )
                        else:
                            final_state_output = None  # Неизвестный формат
                    break  # Завершаем цикл после получения финального состояния графа

        if final_state_output:
            final_messages: list[AnyMessage] = final_state_output.get("messages", [])
            if final_messages and isinstance(final_messages[-1], AIMessage):
                final_response = final_messages[-1].content
                logger.info(
                    f"Агент завершил работу для thread_id: {thread_id}. Ответ: {final_response[:100]}..."
                )

                # Пример: автоматическое добавление в FAQ, если это был вопрос про Яндекс и ответ успешный
                # Эту логику можно усложнить или вынести
                if "яндекс" in text.lower() and not any(
                    err in final_response.lower()
                    for err in ["ошибка", "не удалось", "не смог"]
                ):
                    try:
                        async with asyncio.timeout(5):  # Таймаут для операции с БД
                            # Используем run_in_executor для синхронных вызовов БД внутри async функции
                            loop = asyncio.get_running_loop()
                            await loop.run_in_executor(
                                None, _add_to_faq_if_new, text, final_response
                            )
                    except asyncio.TimeoutError:
                        logger.error("Таймаут при добавлении в FAQ")
                    except Exception as db_err:
                        logger.error(
                            f"Ошибка при попытке добавить запись в FAQ: {db_err}",
                            exc_info=True,
                        )
            else:  # Если нет AIMessage в конце или final_messages пуст
                logger.warning(
                    f"Агент завершился, но не найдено финального AIMessage для thread_id: {thread_id}"
                )
                # Попытка найти последний AIMessage в истории, если он там есть
                ai_messages = [m for m in final_messages if isinstance(m, AIMessage)]
                if ai_messages:
                    final_response = ai_messages[-1].content
                    logger.info(
                        f"Извлечен последний AIMessage из истории: {final_response[:100]}..."
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

    except asyncio.TimeoutError:
        logger.error(f"Превышено время ожидания ответа от агента для user {user_id}")
        final_response = "Извините, обработка запроса заняла слишком много времени. Пожалуйста, попробуйте еще раз."
    except Exception as e:
        logger.error(
            f"Ошибка при вызове агента для user {user_id} в чате {chat_id}: {e}",
            exc_info=True,
        )
        final_response = (
            "Извините, произошла внутренняя ошибка при обработке вашего запроса."
        )

    try:
        await context.bot.send_message(
            chat_id=chat_id, text=final_response, parse_mode=ParseMode.MARKDOWN
        )
    except Exception as send_error:  # Более конкретное имя для ошибки отправки
        logger.error(
            f"Ошибка при отправке ответа в чат {chat_id}: {send_error}", exc_info=True
        )
        # Попытка отправить без форматирования, если Markdown вызвал ошибку
        try:
            plain_text_response = f"Произошла ошибка при форматировании ответа. Ответ без форматирования:\n\n{final_response}"
            await context.bot.send_message(chat_id=chat_id, text=plain_text_response)
        except Exception as e2:
            logger.error(
                f"Не удалось отправить ответ даже без форматирования: {e2}",
                exc_info=True,
            )


def _add_to_faq_if_new(text: str, final_response: str):
    """Вспомогательная синхронная функция для работы с БД FAQ."""
    try:
        with connection.get_db_session() as db:
            if db:
                similar = search_faq_entries(db, text, limit=1, min_similarity=0.95)
                if not similar:
                    logger.info(
                        f"Добавляю новый вопрос про Яндекс в FAQ: {text[:50]}..."
                    )
                    add_faq_entry(db, question=text, answer=final_response)
                else:
                    logger.info(
                        "Похожий вопрос про Яндекс уже есть в FAQ, не добавляю."
                    )
            else:
                logger.error(
                    "Не удалось получить сессию БД для добавления в FAQ (синхронная функция)."
                )
    except Exception as e:
        # Логируем ошибку здесь, т.к. run_in_executor не пробросит ее наверх легко
        logger.error(f"Ошибка в _add_to_faq_if_new: {e}", exc_info=True)
