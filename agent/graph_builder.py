import logging
import os
from typing import Optional, Tuple
import aiosqlite

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# --- Импорт узлов и настройки ---
from .graph_nodes import (
    input_guardrails_node,
    analyze_context,
    should_respond,
    router_node,
    tool_executor_node,
    tool_output_guardrails_node,
    handle_tool_result_node,
    response_generator_node,
    output_guardrails_node,
    set_node_llm,
)
from .llm_setup import setup_llm
from .state import AgentState

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """Строит граф LangGraph для QA-агента."""
    graph = StateGraph(AgentState)

    # Добавление узлов
    graph.add_node("input_guardrails", input_guardrails_node)
    graph.add_node("analyze_context", analyze_context)
    graph.add_node("router", router_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("tool_output_guardrails", tool_output_guardrails_node)
    graph.add_node("response_generator", response_generator_node)
    graph.add_node("output_guardrails", output_guardrails_node)

    # Определение рёбер
    graph.set_entry_point("input_guardrails")

    graph.add_edge("input_guardrails", "analyze_context")
    graph.add_edge("analyze_context", "router")

    # Условный переход после роутера
    graph.add_conditional_edges(
        "router",
        # Функция определяет, был ли вызван инструмент
        lambda state: (
            "tool_executor"
            if (
                isinstance(state["messages"][-1], AIMessage)
                and state["messages"][-1].tool_calls
            )
            else "response_generator"
        ),
        {
            "tool_executor": "tool_executor",
            "response_generator": "response_generator",
        },
    )

    graph.add_edge("tool_executor", "tool_output_guardrails")
    graph.add_edge("tool_output_guardrails", "response_generator")

    # После генерации ответа и его проверки всегда завершаем цикл
    graph.add_edge("response_generator", "output_guardrails")
    graph.add_edge("output_guardrails", END)

    logger.info("Граф LangGraph успешно построен.")
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


# Импорт AIMessage для типизации
from langchain_core.messages import AIMessage
