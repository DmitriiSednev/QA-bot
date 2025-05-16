import logging
import os
from typing import Literal, Optional, Tuple, List
import aiosqlite

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage, SystemMessage

# --- Импорт узлов и настройки --- #
# Обновляем импорты узлов из новых файлов
from .input_output_nodes import (
    input_guardrails_node,
    response_generator_node,
    output_guardrails_node,
)
from .routing_and_tool_nodes import (
    # analyze_context, # analyze_context не используется напрямую в build_graph
    router_node,
    tool_executor_node,
    tool_output_guardrails_node,
    # should_respond, # should_respond не используется как узел, а как условная функция (если будет)
)
from .node_utils import set_node_llm  # set_node_llm и trim_messages

from .llm_setup import setup_llm
from .state import AgentState

logger = logging.getLogger(__name__)


def router_condition(
    state: AgentState,
) -> Literal["tools", "output_guardrails", "__end__"]:
    """
    Условная функция, определяющая следующий узел после router_node.
    Направляет на 'tools', если AIMessage от роутера содержит tool_calls.
    Направляет на 'output_guardrails', если AIMessage от роутера НЕ содержит tool_calls (прямой ответ или ошибка).
    """
    logger.debug("--- Условное ребро: router_condition ---")
    messages: List[BaseMessage] = state["messages"]
    if not messages:
        logger.warning("router_condition: Нет сообщений в состоянии, завершение графа.")
        return "__end__"

    last_message = messages[-1]
    logger.debug(
        f"router_condition: Анализ сообщения от router_node: тип {type(last_message)}, content: '{str(last_message.content)[:100]}...', tool_calls: {hasattr(last_message, 'tool_calls') and last_message.tool_calls is not None}"
    )

    if not isinstance(last_message, AIMessage):
        logger.error(
            f"router_condition: Ожидалось AIMessage от роутера, но получено {type(last_message)}. "
            "Это указывает на проблему в графе или самом router_node. Завершение."
        )
        # Можно добавить AIMessage с ошибкой в состояние перед завершением, если это поможет отладке
        # state["messages"] = add_messages(state["messages"], [AIMessage(content="Ошибка графа: router_condition ожидал AIMessage.")])
        return "__end__"

    if last_message.tool_calls:
        logger.info(
            "router_condition: AIMessage от роутера содержит tool_calls. Переход к 'tools'."
        )
        return "tools"
    else:
        logger.info(
            "router_condition: AIMessage от роутера НЕ содержит tool_calls (прямой ответ или ошибка). Переход к 'output_guardrails'."
        )
        return "output_guardrails"


def trim_messages(messages, max_history=5):
    # Убираем системные сообщения, если есть
    filtered = [m for m in messages if not isinstance(m, SystemMessage)]
    return filtered[-max_history:]


def build_graph() -> StateGraph:
    """Собирает и возвращает граф LangGraph."""
    logger.info("Сборка графа агента...")

    # Инициализация LLM при сборке графа
    llm = setup_llm()
    if not llm:
        logger.critical("Не удалось инициализировать LLM. Граф не может быть собран.")
        raise ValueError("LLM initialization failed.")
    set_node_llm(llm)  # Передаем LLM в узлы

    graph = StateGraph(AgentState)

    # --- Добавление узлов --- #
    graph.add_node("input_guardrails", input_guardrails_node)
    graph.add_node("router", router_node)
    graph.add_node("tools", tool_executor_node)
    graph.add_node("tool_output_guardrails", tool_output_guardrails_node)
    # graph.add_node("handle_tool_result", handle_tool_result_node) # Уже удален
    graph.add_node("response_generator", response_generator_node)
    graph.add_node("output_guardrails", output_guardrails_node)

    # --- Определение точки входа --- #
    graph.set_entry_point("input_guardrails")

    # --- Определение ребер --- #
    graph.add_edge(
        "input_guardrails", "router"
    )  # После проверки ввода всегда к роутеру

    # Условное ребро после роутера
    graph.add_conditional_edges(
        "router",
        router_condition,  # Используем новую функцию router_condition
        {
            "tools": "tools",
            "output_guardrails": "output_guardrails",  # Прямой ответ/ошибка от роутера идет на финальные проверки
            # "response_generator" больше не является прямым выходом из router_condition
            "__end__": "__end__",
        },
    )

    graph.add_edge(
        "tools", "tool_output_guardrails"
    )  # После инструментов всегда к их проверке
    graph.add_edge(
        "tool_output_guardrails", "response_generator"
    )  # После проверки вывода инструмента сразу к генератору ответа

    graph.add_edge("response_generator", "output_guardrails")
    graph.add_edge("output_guardrails", END)  # После финальной проверки - конец

    logger.info("Граф успешно собран.")
    return graph


async def setup_agent() -> Optional[Tuple[StateGraph, aiosqlite.Connection]]:
    """Инициализирует и компилирует агент LangGraph."""
    logger.info("Инициализация агента LangGraph...")

    # --- Инициализация LLM --- #
    llm = setup_llm()
    if not llm:
        return None

    # Передаем LLM в модуль с узлами
    set_node_llm(llm)

    # --- Хранилище состояний (Memory) --- #
    os.makedirs("checkpoints", exist_ok=True)
    db_path = "checkpoints/langgraph_agent.sqlite"
    checkpoint_conn: Optional[aiosqlite.Connection] = None
    try:
        # Создаем соединение aiosqlite напрямую
        checkpoint_conn = await aiosqlite.connect(db_path)
        logger.info(f"aiosqlite соединение для checkpointer успешно открыто: {db_path}")
    except aiosqlite.Error as e:
        logger.critical(
            f"Не удалось подключиться к aiosqlite для checkpointer ({db_path}): {e}",
            exc_info=True,
        )
        return None

    # Инициализируем AsyncSqliteSaver с созданным соединением
    memory = AsyncSqliteSaver(conn=checkpoint_conn)
    logger.info("AsyncSqliteSaver (checkpointer) настроен.")

    # --- Построение и компиляция графа --- #
    try:
        graph = build_graph()
        # Компилируем граф с памятью
        agent_app = graph.compile(checkpointer=memory)
        logger.info("Агент LangGraph успешно скомпилирован с хранилищем состояний.")
        return agent_app, checkpoint_conn
    except Exception as e:
        logger.critical(
            f"Ошибка при построении или компиляции графа: {e}", exc_info=True
        )
        if checkpoint_conn:
            try:
                await checkpoint_conn.close()
                logger.info(
                    "Соединение aiosqlite для checkpointer закрыто из-за ошибки компиляции графа."
                )
            except aiosqlite.Error as ce:
                logger.error(
                    f"Ошибка при закрытии соединения aiosqlite для checkpointer: {ce}",
                    exc_info=True,
                )
        return None
