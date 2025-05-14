import logging
import os
from typing import Literal, Optional, Tuple, List
import aiosqlite

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage, SystemMessage

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


def should_continue(
    state: AgentState,
) -> Literal["tools", "response_generator", "__end__"]:
    """Определяет следующий шаг после роутера или проверки ввода."""
    logger.debug("--- Условное ребро: should_continue ---")
    messages: List[BaseMessage] = state["messages"]
    if not messages:
        logger.warning("should_continue: Нет сообщений, завершение.")
        return "__end__"

    last_message = messages[-1]
    logger.debug(
        f"should_continue: Последнее сообщение: {type(last_message)}: {str(last_message)[:200]}"
    )

    # Если input_guardrails добавил AIMessage с ошибкой (например, пустой ввод)
    if (
        isinstance(last_message, AIMessage)
        and last_message.content.startswith("Ошибка")
        or last_message.content == "Пожалуйста, введите ваш вопрос."
    ):
        logger.info(
            "should_continue: Обнаружено сообщение об ошибке от input_guardrails. Завершение."
        )
        return "__end__"  # Или можно направить на output_guardrails, если он должен форматировать такие ошибки

    # Если router_node вернул AIMessage с tool_calls
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.info(
            "should_continue: Router выбрал инструмент. Переход к tool_executor."
        )
        return "tools"

    # Если router_node вернул AIMessage без tool_calls (прямой ответ) или
    # если router_node вернул ошибку (которая будет AIMessage)
    # или если это результат от input_guardrails (пройденный)
    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
        logger.info(
            "should_continue: Router сгенерировал прямой ответ или вернул ошибку. Переход к response_generator (или output_guardrails)."
        )
        # Если router вернул ошибку, response_generator должен ее обработать или передать.
        # Если router дал прямой ответ, response_generator может его доработать или нет, в зависимости от промпта.
        # В нашей текущей логике router_node, если нет tool_calls, то AIMessage уже содержит финальный ответ,
        # либо сообщение об ошибке. В этом случае, response_generator не нужен, можно сразу идти к output_guardrails и END.
        # НО! Если мы хотим, чтобы response_generator всегда вызывался для форматирования/дополнения, то оставляем так.
        # Пока что направим на response_generator, который должен быть готов к таким случаям.
        return "response_generator"

    # Если последнее сообщение - это HumanMessage (после input_guardrails, перед router)
    # или если это результат от tool_executor (ToolMessage), который прошел guardrails и handle_tool_result
    # В этих случаях router должен быть следующим.
    # Однако, у нас есть прямой переход от handle_tool_result к response_generator
    # Этот путь (возврат к router после инструмента) сейчас не активен в графе.
    logger.debug(
        "should_continue: Не AIMessage с tool_calls и не прямой ответ/ошибка от router. Решение по умолчанию - к response_generator."
    )
    # Это может быть состояние после input_guardrails (HumanMessage) - тогда граф пойдет к router.
    # Или после tool_executor -> tool_output_guardrails -> handle_tool_result (ToolMessage)
    #  -> response_generator (логика из agent_executor)
    # Если мы здесь после handle_tool_result, то это ToolMessage, и мы должны идти в response_generator
    if isinstance(last_message, ToolMessage):
        logger.info(
            "should_continue: Последнее сообщение - ToolMessage. Переход к response_generator."
        )
        return "response_generator"

    logger.warning(
        f"should_continue: Неожиданное состояние, last_message: {last_message}. Завершение графа."
    )
    return "__end__"


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
    graph.add_node(
        "handle_tool_result", handle_tool_result_node
    )  # Оставляем, но можно упростить/удалить
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
        should_continue,  # Эта функция решает, куда идти: к инструментам или к генератору
        {
            "tools": "tools",
            "response_generator": "response_generator",
            "__end__": "__end__",  # Если should_continue вернет "__end__"
        },
    )

    graph.add_edge(
        "tools", "tool_output_guardrails"
    )  # После инструментов всегда к их проверке
    graph.add_edge(
        "tool_output_guardrails", "handle_tool_result"
    )  # После проверки к обработчику результата

    # Условное ребро после handle_tool_result (по вашей логике agent_executor, после инструмента всегда к генератору)
    # В текущей реализации should_continue и handle_tool_result, этот путь такой:
    # handle_tool_result -> (ничего не меняет) -> should_continue (видит ToolMessage) -> response_generator
    # Это немного избыточно. Можно сделать прямой переход или упростить should_continue.
    # Пока оставляем так, как ближе к вашей исходной логике, где после tool_result всегда response_generator
    graph.add_edge("handle_tool_result", "response_generator")
    # TODO: Рассмотреть упрощение: tool_output_guardrails -> (условное ребро) -> response_generator / END
    # Это убрало бы handle_tool_result и сделало бы should_continue проще.

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
