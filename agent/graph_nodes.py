import logging
import json
from typing import Annotated, Optional, List, Dict, Any, Sequence, Tuple, Union, Literal
import operator

from langgraph.graph import END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    FunctionMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatYandexGPT

# --- Импорт состояния и реестра инструментов ---
from .state import AgentState
from .tools_registry import (
    tool_registry,
    tools_list,
)  # Используем tools_list для llm.bind_tools

# --- Настройка логгера ---
logger = logging.getLogger(__name__)

# --- Глобальная переменная для LLM (устанавливается в llm_setup.py) ---
llm = None


def set_node_llm(node_llm):
    """Устанавливает LLM для использования в узлах."""
    global llm
    llm = node_llm


# --- Узлы графа --- #
def input_guardrails_node(state):
    """Проверяет входящее сообщение пользователя."""
    logger.debug("===Input Guardrails===")
    messages = state["messages"]
    if not messages or not isinstance(messages[-1], HumanMessage):
        logger.warning("Input Guardrails: Последнее сообщение не от пользователя.")
        return {
            "messages": messages
            + [AIMessage(content="Ошибка: Ожидалось сообщение от пользователя.")]
        }

    # Простые проверки (можно добавить более сложные: токсичность, PII и т.д.)
    user_input = messages[-1].content
    if not user_input.strip():
        logger.warning("Input Guardrails: Пустое сообщение от пользователя.")
        return {
            "messages": messages
            + [AIMessage(content="Пожалуйста, введите ваш вопрос.")]
        }

    logger.debug("Input Guardrails: Проверка пройдена.")
    return {}


def analyze_context(state):
    """Анализирует текущий контекст диалога."""
    logger.debug("===Analyze Context===")
    # Здесь может быть логика анализа истории, извлечения сущностей и т.д.
    # Можно использовать инструмент context_analyzer
    context_analysis_result = "Контекст проанализирован."
    logger.debug(f"Context Analysis Result: {context_analysis_result}")
    # Можно добавить результат анализа в состояние, если он нужен дальше
    # return {"context_analysis": context_analysis_result}
    return {}


def should_respond(state) -> Literal["continue", "__end__"]:
    """Проверяет, нужно ли продолжать или можно закончить."""
    logger.debug("===Checking Should Respond===")
    messages = state["messages"]
    # Если последнее сообщение - ответ от LLM без запроса инструментов, то конец
    if isinstance(messages[-1], AIMessage) and not messages[-1].tool_calls:
        logger.debug(
            "Should Respond: Last message is AIMessage without tool calls. ENDING."
        )
        return END
    # Если есть ошибка в сообщениях (кроме ошибок вызова инструментов)
    if any(isinstance(m, AIMessage) and "Ошибка:" in m.content for m in messages):
        # Исключаем ошибки, которые могут быть обработаны в handle_tool_result_node
        if not isinstance(messages[-1], ToolMessage) and not (
            isinstance(messages[-1], AIMessage) and messages[-1].tool_calls
        ):
            logger.warning("Should Respond: Found error message. ENDING.")
            return END
    # Если вызов инструмента завершился ошибкой (и мы ее уже показали пользователю)
    if isinstance(messages[-1], ToolMessage) and messages[-1].additional_kwargs.get(
        "is_error", False
    ):
        logger.debug("Should Respond: Last message is ToolMessage with error. ENDING.")
        return END

    logger.debug("Should Respond: Continuing.")
    return "continue"


# --- Роутер: Выбор между вызовом инструмента и генерацией ответа --- #
async def router_node(state):
    """Маршрутизатор: решает, вызывать инструмент или генерировать ответ."""
    logger.debug("===Router Node===")
    if not llm:
        logger.error("Router Node: LLM не инициализирована!")
        return {
            "messages": add_messages(
                state["messages"], [AIMessage(content="Ошибка: LLM не настроена.")]
            )
        }

    # Привязываем инструменты к LLM для выбора
    llm_with_tools = llm.bind_tools(tools_list)

    # Формируем промпт для роутера
    prompt_text = """Ты умный ИИ-ассистент. Проанализируй последний вопрос пользователя и историю диалога.
**Важно:** Если вопрос касается **Yandex Cloud**, его сервисов, API, CLI и т.д., **ВСЕГДА сначала используй инструмент `search_yandex_documentation`**.
Если вопрос общего характера или касается внутренней базы знаний, используй `search_faq`.
Если нужно выполнить CRUD операции с FAQ или поискать в истории, используй соответствующие инструменты.
Если информации достаточно и инструмент не нужен, просто ответь пользователю.
Доступные инструменты: {[t.name for t in tools_list]}"""

    router_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_text),
            MessagesPlaceholder(variable_name="messages", optional=True),
        ]
    )

    # Цепочка для вызова роутера
    chain = router_prompt | llm_with_tools

    try:
        # Передаем только последние N сообщений, чтобы не превышать лимит контекста
        MAX_ROUTER_MESSAGES = 10
        routing_messages = state["messages"][-MAX_ROUTER_MESSAGES:]

        # Убираем системные сообщения
        routing_messages = [m for m in routing_messages if m.type != "system"]

        ai_message = await chain.ainvoke({"messages": routing_messages})
        logger.debug(f"Router Node: LLM response: {ai_message}")

        # Добавляем ответ LLM в состояние
        return {"messages": add_messages(state["messages"], [ai_message])}

    except Exception as e:
        logger.error(
            f"Router Node: Ошибка при вызове LLM для маршрутизации: {e}", exc_info=True
        )
        # Возвращаем сообщение об ошибке
        error_message = AIMessage(content=f"Ошибка при принятии решения: {e}")
        return {"messages": add_messages(state["messages"], [error_message])}


# --- Узел выполнения инструментов --- #
def tool_executor_node(state):
    """Выполняет вызванные инструменты."""
    logger.debug("===Tool Executor Node===")
    messages = state["messages"]
    last_message = messages[-1]

    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        logger.debug("Tool Executor Node: Нет вызовов инструментов.")
        return {}

    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        logger.info(
            f"Tool Executor Node: Подготовка к вызову инструмента '{tool_name}' с аргументами: {tool_args}"
        )

        # Найти инструмент по имени
        tool = next((t for t in tools_list if t.name == tool_name), None)

        if tool is None:
            is_error = True
            content = f"Ошибка: инструмент '{tool_name}' не найден."
        else:
            try:
                # Напрямую вызываем инструмент
                content = tool.run(tool_args)
                is_error = False
            except Exception as e:
                content = f"Ошибка выполнения инструмента: {e}"
                is_error = True

        logger.info(
            f"Tool Executor Node: Результат '{tool_name}': {content[:200]}...{' (Ошибка)' if is_error else ''}"
        )

        tool_messages.append(
            ToolMessage(
                tool_call_id=tool_call["id"],
                content=content,
                additional_kwargs={"is_error": is_error},
            )
        )

    return {"messages": tool_messages}


# --- Узел обработки результатов инструментов (Guardrails) --- #
def tool_output_guardrails_node(state):
    """Проверяет вывод инструментов."""
    logger.debug("===Tool Output Guardrails===")
    messages = state["messages"]
    last_message = messages[-1]

    if not isinstance(last_message, ToolMessage):
        logger.debug("Tool Output Guardrails: Последнее сообщение не от инструмента.")
        return {}

    # Проверка на ошибки
    if last_message.additional_kwargs.get("is_error", False):
        logger.warning(
            f"Tool Output Guardrails: Обнаружена ошибка в выводе инструмента (ID: {last_message.tool_call_id}): {last_message.content}"
        )
    else:
        logger.debug(
            f"Tool Output Guardrails: Вывод инструмента (ID: {last_message.tool_call_id}) проверен."
        )

    return {}


# --- Узел для решения, что делать после инструмента --- #
def handle_tool_result_node(state) -> Literal["generate_response", "__end__"]:
    """Решает, генерировать ли ответ после инструмента или закончить."""
    logger.debug("===Handle Tool Result Node===")
    messages = state["messages"]
    last_message = messages[-1]

    # Если последний вызов инструмента вернул ошибку, завершаем
    if isinstance(last_message, ToolMessage) and last_message.additional_kwargs.get(
        "is_error", False
    ):
        logger.warning("Handle Tool Result: Обнаружена ошибка инструмента. Завершение.")
        return END

    # Если все инструменты отработали без ошибок, генерируем ответ
    logger.debug(
        "Handle Tool Result: Инструменты отработали успешно. Переход к генерации ответа."
    )
    return "generate_response"


# --- Узел генерации ответа --- #
async def response_generator_node(state):
    """Генерирует финальный ответ пользователю."""
    logger.debug("===Response Generator Node===")
    if not llm:
        logger.error("Response Generator: LLM не инициализирована!")
        return {
            "messages": add_messages(
                state["messages"], [AIMessage(content="Ошибка: LLM не настроена.")]
            )
        }

    # --- Системный промпт для генерации ответа --- #
    system_prompt = """Ты - дружелюбный и компетентный ИИ-ассистент QA-бота для команды поддержки Yandex Cloud.

Твоя задача - отвечать на вопросы пользователей, используя предоставленную информацию.

**Приоритет источников информации:**
1.  **Результаты поиска по документации Yandex Cloud** (из инструмента `search_yandex_documentation`). Это самый важный источник для вопросов о Yandex Cloud.
2.  **Результаты поиска по базе знаний FAQ** (из инструмента `search_faq`). Используй это для специфичных вопросов проекта или часто задаваемых вопросов.
3.  **Результаты поиска по истории чата** (из инструмента `search_chat_history`). Может помочь найти похожие решенные проблемы.
4.  **Результат анализа контекста** (если есть).
5.  Твои общие знания (используй их, только если информация из инструментов недоступна или нерелевантна).

**Правила ответа:**

-   Отвечай **на русском языке**.
-   Будь вежливым и профессиональным.
-   Если информация найдена в документации Yandex Cloud или FAQ, **дай прямой и четкий ответ**, основанный на этой информации. Можешь кратко переформулировать, но не добавляй от себя лишнего.
-   Если релевантная информация найдена в истории чата, кратко упомяни, что похожий вопрос обсуждался, и изложи суть найденного ответа.
-   Если инструменты вернули сообщение об ошибке или отсутствии информации, сообщи пользователю, что найти точный ответ не удалось. Не придумывай ответ.
-   **Не упоминай названия инструментов**, которые ты использовал (например, не пиши 'Согласно search_faq...'). Просто предоставляй информацию.
-   Если пользователь просил добавить/изменить/удалить запись в FAQ, подтверди выполнение операции (сообщение об этом будет в выводе соответствующего инструмента).
-   Форматируй код и команды с помощью Markdown (```).
-   Ответ должен быть **одним сообщением**.

Текущий диалог:
"""

    # Формируем промпт для генератора
    generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    chain = generator_prompt | llm

    try:
        # Убираем сообщения с ошибками инструментов из контекста для LLM
        messages_for_llm = [
            m
            for m in state["messages"]
            if not (
                isinstance(m, ToolMessage)
                and m.additional_kwargs.get("is_error", False)
            )
        ]
        # Ограничиваем количество сообщений для LLM
        MAX_GENERATOR_MESSAGES = 15
        messages_for_llm = messages_for_llm[-MAX_GENERATOR_MESSAGES:]

        response_message = await chain.ainvoke({"messages": messages_for_llm})
        logger.debug(f"Response Generator: LLM response: {response_message}")

        # Добавляем финальный ответ в состояние
        return {"messages": add_messages(state["messages"], [response_message])}

    except Exception as e:
        logger.error(
            f"Response Generator: Ошибка при генерации ответа: {e}", exc_info=True
        )
        # Возвращаем сообщение об ошибке
        error_message = AIMessage(content=f"Ошибка при генерации ответа: {e}")
        return {"messages": add_messages(state["messages"], [error_message])}


# --- Узел выходных проверок --- #
def output_guardrails_node(state):
    """Проверяет финальный ответ перед отправкой пользователю."""
    logger.debug("===Output Guardrails===")
    messages = state["messages"]
    last_message = messages[-1]

    if not isinstance(last_message, AIMessage):
        logger.warning("Output Guardrails: Финальное сообщение не AIMessage.")
        # Можно заменить на стандартный ответ или оставить как есть
        return {}

    # Проверка на пустой ответ
    if not last_message.content.strip():
        logger.warning("Output Guardrails: Финальный ответ пустой.")
        # Заменяем на стандартный ответ
        fallback_response = AIMessage(content="Извините, я не смог сформировать ответ.")
        return {"messages": messages[:-1] + [fallback_response]}

    # Другие проверки (например, на наличие запрещенного контента)

    logger.debug("Output Guardrails: Финальный ответ проверен.")
    return {}
