import logging
import json
from typing import Dict, Any, List
import asyncio

from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .state import AgentState
from .node_utils import llm_instance, trim_messages, LLM_TIMEOUT # Импорт из node_utils

logger = logging.getLogger(__name__)

# Системный промпт для генератора ответа
SYSTEM_PROMPT_RESPONSE_GENERATOR = """Ты - полезный AI ассистент. Отвечай на последний вопрос пользователя ясно и по делу,
учитывая всю предыдущую историю диалога, включая результаты вызова инструментов (ToolMessage).
Основывай свой ответ на результатах инструментов, если они релевантны.
Если последний запрос был на выполнение действия (add/update/delete) и он выполнен успешно (видно из ToolMessage),
просто подтверди это кратко.
"""

# --- Узлы графа --- #
def input_guardrails_node(state: AgentState) -> Dict[str, Any]:
    """Проверяет входящее сообщение пользователя."""
    logger.info("--- Вход в Input Guardrails ---")
    messages = state["messages"]
    if not messages or not isinstance(messages[-1], HumanMessage):
        logger.warning(
            "Input Guardrails: Последнее сообщение не от пользователя или пусто."
        )
        error_ai_msg = AIMessage(
            content="Ошибка: Ожидалось непустое сообщение от пользователя."
        )
        return {"messages": add_messages(messages, [error_ai_msg])}

    user_input = messages[-1].content
    if not user_input.strip():
        logger.warning("Input Guardrails: Пустое сообщение от пользователя.")
        error_ai_msg = AIMessage(content="Пожалуйста, введите ваш вопрос.")
        return {"messages": add_messages(messages, [error_ai_msg])}

    logger.debug("Input Guardrails: Проверка пройдена.")
    return {} 

async def response_generator_node(state: AgentState) -> Dict[str, Any]:
    """Генерирует финальный ответ пользователю."""
    logger.info("--- Вход в Генератор Ответа ---")
    if not llm_instance:
        logger.error("Response Generator: LLM не инициализирована!")
        error_ai_msg = AIMessage(
            content="Ошибка конфигурации: LLM не настроена для генерации ответа."
        )
        return {"messages": add_messages(state["messages"], [error_ai_msg])}

    trimmed_messages = trim_messages(state["messages"], max_history=5)
    messages_for_llm = [
        SystemMessage(content=SYSTEM_PROMPT_RESPONSE_GENERATOR),
        *trimmed_messages,
    ]

    logger.debug(f"Сообщения для генератора ответа LLM: {messages_for_llm}")
    try:
        try:
            payload_size = len(
                json.dumps(
                    [
                        m.dict() if hasattr(m, "dict") else str(m)
                        for m in messages_for_llm
                    ]
                )
            )
            logger.info(
                f"Payload size before LLM call (generator): {payload_size} bytes, messages count: {len(messages_for_llm)}"
            )
        except Exception as log_e:
            logger.warning(
                f"Ошибка при логировании размера payload в response_generator_node: {log_e}"
            )

        async with asyncio.timeout(LLM_TIMEOUT):
            ai_response: AIMessage = await llm_instance.ainvoke(messages_for_llm)

        logger.info(f"Сгенерированный ответ: {ai_response.content[:200]}...")
        return {"messages": add_messages(state["messages"], [ai_response])}
    except asyncio.TimeoutError:
        logger.error(f"Таймаут ({LLM_TIMEOUT} сек) при вызове LLM-генератора.")
        error_ai_msg = AIMessage(
            content="Извините, генератор ответа LLM не ответил вовремя."
        )
        return {"messages": add_messages(state["messages"], [error_ai_msg])}
    except Exception as e:
        logger.error(f"Ошибка при генерации ответа LLM: {e}", exc_info=True)
        error_ai_msg = AIMessage(
            content=f"Извините, произошла ошибка при генерации ответа: {e}"
        )
        return {"messages": add_messages(state["messages"], [error_ai_msg])}

def output_guardrails_node(state: AgentState) -> Dict[str, Any]:
    """Проверяет финальный ответ перед отправкой пользователю."""
    logger.info("--- Вход в Output Guardrails ---")
    messages = state["messages"]
    if not messages or not isinstance(messages[-1], AIMessage):
        logger.warning(
            "Output Guardrails: Последнее сообщение не AIMessage или пусто. Нечего проверять."
        )
        return {}

    last_ai_message = messages[-1]
    if not last_ai_message.content or not last_ai_message.content.strip():
        logger.warning(
            "Output Guardrails: Финальный ответ AIMessage пустой. Замена на fallback."
        )
        fallback_response = AIMessage(
            content="Извините, я не смог сформировать внятный ответ на ваш запрос."
        )
        updated_messages = list(state["messages"][:-1]) + [fallback_response]
        return {"messages": updated_messages}

    logger.debug(
        f"Output Guardrails: Финальный ответ проверен: {last_ai_message.content[:100]}..."
    )
    return {} 