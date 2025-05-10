import logging
import operator
import os
import functools
import uuid  # Для генерации фейкового tool_call_id
from typing import Sequence, Literal, Dict, Any

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# Попытка импортировать YandexGPT, если будет использоваться
try:
    from langchain_community.llms.yandex import YandexGPT

    # ChatYandexGPT может быть еще не в стабильной community,
    # используем YandexGPT, который может работать с чат-логикой через messages.
    # Если есть ChatYandexGPT, лучше использовать его.
    # from langchain_community.chat_models.yandex import ChatYandexGPT
    YANDEX_VERFÜGBAR = True
except ImportError:
    YANDEX_VERFÜGBAR = False
    YandexGPT = None  # чтобы type hinting не ругался

from .state import AgentState

# --- Импорт инструментов ---
from .tools.web_search import WebSearchTool
from .tools.search_faq import SearchFAQTool
from .tools.add_faq import AddFAQTool
from .tools.update_faq import UpdateFAQTool
from .tools.delete_faq import DeleteFAQTool
from agent.tools.tavily_yandexcloud_search import TavilyYandexCloudSearchTool

# Инициализация логгера
logger = logging.getLogger(__name__)

# Системный промпт для генератора ответа
SYSTEM_PROMPT_RESPONSE_GENERATOR = """Ты - полезный AI ассистент. Отвечай на последний вопрос пользователя ясно и по делу,
учитывая всю предыдущую историю диалога, включая результаты вызова инструментов (ToolMessage).
Основывай свой ответ на результатах инструментов, если они релевантны.
Если последний запрос был на выполнение действия (add/update/delete) и он выполнен успешно (видно из ToolMessage),
просто подтверди это кратко.
"""

# Флаг для тестирования только Tavily API без вызовов LLM
TEST_TAVILY_ONLY_MODE = False  # <--- ИЗМЕНЕНО ДЛЯ ПОЛНОГО ТЕСТА

# Реестр реальных объектов инструментов
tool_registry: Dict[str, BaseTool] = {}

# COMMAND_TOOLS = {"add_faq", "update_faq", "delete_faq"}


def input_guardrails_node(state: AgentState):
    logger.info("--- Вход в Input Guardrails ---")
    # TODO: Реализовать логику input guardrails (например, проверка на PII, токсичность)
    # Пока что просто пропускаем
    return {}


# ---- НОВЫЙ УМНЫЙ РОУТЕР ----
def router_node(state: AgentState, llm: ChatOpenAI):
    """Анализирует сообщение пользователя и решает, какой инструмент вызвать или генерировать ответ."""
    logger.info("--- Вход в Умный Роутер ---")
    messages = state["messages"]
    last_message = messages[-1]
    logger.info(f"Последнее сообщение для роутера: {last_message.content}")

    if TEST_TAVILY_ONLY_MODE:
        logger.warning("!!! РЕЖИМ ТЕСТИРОВАНИЯ TAVILY API ВКЛЮЧЕН !!!")
        # Принудительно выбираем Tavily и формируем фейковый AIMessage
        tool_name_to_test = (
            TavilyYandexCloudSearchTool().name
        )  # Или 'web_search' если хотите его
        # Убедимся, что инструмент есть в реестре, иначе тест бессмысленный
        if tool_name_to_test not in tool_registry:
            logger.error(
                f"Инструмент '{tool_name_to_test}' не найден в реестре. Тест Tavily невозможен."
            )
            # В реальном коде здесь можно было бы вернуть ошибку или перейти к генерации ответа
            # Для теста просто логируем и позволяем графу идти дальше (скорее всего, будет ошибка в executor)
            return {
                "next_node": "generate_response",
                "messages": messages
                + [
                    AIMessage(
                        content=f"Ошибка: тестовый инструмент {tool_name_to_test} не зарегистрирован."
                    )
                ],
            }

        fake_tool_call_id = f"call_test_tavily_{uuid.uuid4()}"
        fake_ai_message = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": fake_tool_call_id,
                    "name": tool_name_to_test,
                    "args": {
                        "query": last_message.content
                    },  # Используем исходный запрос пользователя
                }
            ],
        )
        logger.info(f"Фейковый AIMessage для теста Tavily: {fake_ai_message}")
        updated_messages = messages + [fake_ai_message]
        return {
            "messages": updated_messages,
            "next_node": "tool_executor",
            "tool_to_call": tool_name_to_test,
            "tool_input": {"query": last_message.content},
            "current_tool_call_id": fake_tool_call_id,
        }

    # --- Стандартная логика роутера (когда TEST_TAVILY_ONLY_MODE = False) ---
    if not tool_registry:
        logger.error(
            "Реестр инструментов (tool_registry) пуст! Невозможно привязать инструменты к LLM."
        )
        error_ai_msg = AIMessage(
            content="Ошибка конфигурации: нет доступных инструментов."
        )
        return {"next_node": "generate_response", "messages": messages + [error_ai_msg]}

    llm_with_tools = llm.bind_tools(list(tool_registry.values()))

    if isinstance(last_message, ToolMessage):
        logger.info(
            "Последнее сообщение - результат инструмента. Переход к генерации ответа."
        )
        # Важно: сообщения уже должны содержать ToolMessage, не нужно его дублировать.
        # next_node указывает графу, что делать дальше.
        return {
            "next_node": "generate_response"
        }  # Сообщения не меняем, они уже в state

    logger.info("Запрос к LLM-роутеру с привязанными инструментами...")
    try:
        ai_message: AIMessage = llm_with_tools.invoke(messages)
        logger.info(f"Ответ LLM-роутера: {ai_message}")

        if not hasattr(ai_message, "tool_calls") or not ai_message.tool_calls:
            logger.info("Router node: LLM не выбрала инструмент.")
            return {
                "next_node": "generate_response",
                "messages": messages
                + [ai_message],  # Добавляем ai_message (без tool_calls) к истории
            }

        # Важно: Добавляем AIMessage с tool_calls в историю ДО того, как передать управление tool_executor'у
        updated_messages = messages + [ai_message]

        tool_call = ai_message.tool_calls[0]
        tool_name = tool_call["name"]
        tool_input = tool_call["args"]
        tool_call_id = tool_call.get("id")

        logger.info(
            f"LLM выбрала инструмент: '{tool_name}' с аргументами: {tool_input}"
        )
        logger.info(f"Router node: Извлеченный tool_call_id: {tool_call_id}")

        if tool_name not in tool_registry:
            logger.error(
                f"LLM выбрала неизвестный инструмент: {tool_name}. Инструменты в реестре: {list(tool_registry.keys())}"
            )
            error_msg_content = (
                f"Ошибка: попытка вызова неизвестного инструмента '{tool_name}'."
            )
            # Формируем ToolMessage с ошибкой, чтобы корректно обработать tool_call_id
            # AIMessage уже добавлен в updated_messages.
            # Теперь добавляем ToolMessage, который соответствует этому AIMessage.
            error_tool_msg = ToolMessage(
                content=error_msg_content,
                tool_call_id=tool_call_id or f"error_unknown_tool_{tool_name}",
            )
            return {
                "messages": updated_messages + [error_tool_msg],
                "next_node": "generate_response",  # После ошибки инструмента идем генерировать ответ
            }

        return {
            "messages": updated_messages,
            "next_node": "tool_executor",
            "tool_to_call": tool_name,
            "tool_input": tool_input,
            "current_tool_call_id": tool_call_id,
        }
    except Exception as e:
        logger.error(
            f"Ошибка в router_node при вызове LLM с инструментами: {e}", exc_info=True
        )
        error_ai_msg = AIMessage(
            content=f"Произошла ошибка при попытке выбрать действие: {e}"
        )
        return {
            "messages": messages + [error_ai_msg],
            "next_node": "generate_response",
        }


# Исполнитель инструментов
def tool_executor_node(state: AgentState):
    """Вызывает выбранный роутером инструмент и передает current_tool_call_id дальше."""
    logger.info("--- Вход в Tool Executor ---")
    tool_name = state.get("tool_to_call")
    tool_input = state.get("tool_input")
    current_tool_call_id = state.get(
        "current_tool_call_id"
    )  # ID вызова, установленный роутером

    logger.info(
        f"Вызов инструмента: {tool_name} с вводом: {tool_input}, ID вызова: {current_tool_call_id}"
    )

    if not tool_name:
        logger.error("Tool Executor: Нет инструмента для вызова.")
        # current_tool_call_id может быть None, если tool_name не был определен
        # Возвращаем результат, который будет преобразован в ToolMessage
        return {
            "tool_result": "Ошибка: инструмент для вызова не определен.",
            "tool_name_executed": "unknown_tool_error",  # Для корректного создания ToolMessage
            "current_tool_call_id": current_tool_call_id
            or "error_no_tool_name_specified",
        }

    tool_to_execute = tool_registry.get(tool_name)
    if not tool_to_execute:
        logger.error(f"Tool Executor: Инструмент '{tool_name}' не найден в реестре.")
        return {
            "tool_result": f"Ошибка: инструмент '{tool_name}' не найден.",
            "tool_name_executed": tool_name,  # Имя инструмента, который пытались вызвать
            "current_tool_call_id": current_tool_call_id
            or f"error_tool_not_found_{tool_name}",
        }

    # Pydantic модели в BaseTool.invoke ожидают словарь.
    # Если tool_input не словарь, попытаемся его адаптировать.
    if not isinstance(tool_input, dict):
        logger.warning(
            f"Tool Executor: tool_input для '{tool_name}' не является словарем ({type(tool_input)}), попытка адаптации."
        )
        # Простая эвристика: если args_schema инструмента ожидает одно поле `query`
        # и tool_input - это строка, используем ее как значение для `query`.
        # Это нужно улучшать для более общего случая.
        if (
            isinstance(tool_input, str)
            and hasattr(tool_to_execute.args_schema, "__fields__")
            and "query" in tool_to_execute.args_schema.__fields__
        ):
            logger.info(
                f"Адаптация tool_input: используем строку как значение для поля 'query' для инструмента '{tool_name}'."
            )
            tool_input = {"query": tool_input}
        else:
            # Если адаптация не удалась, возвращаем ошибку
            logger.error(
                f"Не удалось адаптировать tool_input ({tool_input}) для инструмента '{tool_name}'. Ожидался dict."
            )
            return {
                "tool_result": f"Ошибка: неверный формат входных данных для инструмента '{tool_name}'.",
                "tool_name_executed": tool_name,
                "current_tool_call_id": current_tool_call_id,
            }

    try:
        result = tool_to_execute.invoke(tool_input)
        logger.info(f"Результат инструмента '{tool_name}': {result}")
    except Exception as e:
        logger.error(
            f"Ошибка при выполнении инструмента '{tool_name}': {e}", exc_info=True
        )
        result = f"Ошибка выполнения инструмента {tool_name}: {e}"

    return {
        "tool_result": result,
        "tool_name_executed": tool_name,  # Имя фактически вызванного инструмента
        "current_tool_call_id": current_tool_call_id,  # Передаем ID дальше для handle_tool_result_node
    }


# Проверка вывода инструмента
def tool_output_guardrails_node(state: AgentState):
    """(Заглушка) Проверяет вывод инструмента и передает ключевые поля дальше."""
    logger.info("--- Вход в Tool Output Guardrails ---")
    tool_result = state.get("tool_result")
    tool_name_executed = state.get("tool_name_executed")
    current_tool_call_id = state.get("current_tool_call_id")  # ID из tool_executor_node
    logger.info(
        f"Результат инструмента '{tool_name_executed}' (ID: {current_tool_call_id}) для проверки: {tool_result}"
    )
    # TODO: Реализовать логику проверки вывода

    # Явно передаем нужные поля дальше, чтобы они не потерялись, если этот узел ничего не меняет
    # Иначе LangGraph может их не сохранить, если они не возвращаются явно.
    return {
        "tool_result": tool_result,
        "tool_name_executed": tool_name_executed,
        "current_tool_call_id": current_tool_call_id,
    }


# ---- НОВЫЙ УЗЕЛ для обработки результата инструмента ----
def handle_tool_result_node(state: AgentState):
    """Создает ToolMessage из результата выполнения инструмента и добавляет в историю."""
    logger.info("--- Вход в Handle Tool Result ---")
    tool_result = state.get("tool_result")
    tool_name = state.get("tool_name_executed")  # Имя инструмента, который был выполнен
    tool_call_id = state.get(
        "current_tool_call_id"
    )  # ID, переданный из tool_executor_node

    logger.debug(
        f"Handle Tool Result: Входящее состояние tool_result: '{tool_result}', tool_name: '{tool_name}', tool_call_id: '{tool_call_id}'"
    )

    if tool_name is None or tool_result is None:
        logger.warning(
            "Handle Tool Result: Нет имени инструмента или результата для обработки."
        )
        # Если нет tool_name, то и ToolMessage создавать не очень корректно.
        # Если tool_call_id отсутствует, это серьезная проблема.
        if tool_call_id is None:
            logger.error(
                "КРИТИЧЕСКАЯ ОШИБКА: current_tool_call_id отсутствует в handle_tool_result_node и нет tool_name!"
            )
            # Создаем fallback ToolMessage, чтобы граф не упал, но это сигнализирует о проблеме
            tool_message = ToolMessage(
                content="Ошибка: отсутствует имя инструмента и ID вызова.",
                name="error_tool",
                tool_call_id="missing_tool_call_id_in_handle_tool_result",
            )
            return {
                "messages": [tool_message]
            }  # Добавляем только это сообщение об ошибке
        # Если есть tool_call_id, но нет tool_name (маловероятно, но возможно)
        tool_message = ToolMessage(
            content=str(tool_result),
            name=tool_name or "unknown_tool_if_id_exists",
            tool_call_id=tool_call_id,
        )
        logger.info(
            f"Создано ToolMessage (возможно, с неполными данными): {tool_message}"
        )
        return {"messages": [tool_message]}

    if not tool_call_id:
        # Этого не должно происходить, если current_tool_call_id корректно передается из router -> executor -> guardrails -> сюда.
        logger.error(
            f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось получить tool_call_id для инструмента '{tool_name}' в handle_tool_result_node. Это ошибка в потоке данных графа."
        )
        # Пытаемся найти последний AIMessage с tool_calls и использовать его ID, если имя совпадает.
        # Это ОЧЕНЬ плохой fallback и указывает на серьезную проблему в логике графа.
        fallback_tool_call_id = f"error_missing_tool_call_id_for_{tool_name}"
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.get("name") == tool_name:
                        fallback_tool_call_id = tc.get("id", fallback_tool_call_id)
                        logger.warning(
                            f"Использован fallback tool_call_id: '{fallback_tool_call_id}' для инструмента '{tool_name}' из истории AIMessages."
                        )
                        break
                if (
                    fallback_tool_call_id
                    != f"error_missing_tool_call_id_for_{tool_name}"
                ):
                    break
        tool_call_id = fallback_tool_call_id

    tool_message = ToolMessage(
        content=str(tool_result), name=tool_name, tool_call_id=tool_call_id
    )
    logger.info(f"Создано ToolMessage: {tool_message}")

    # LangGraph автоматически добавляет возвращаемые сообщения к ключу `messages` в AgentState,
    # если ключ `messages` использует `operator.add`.
    # Поэтому мы возвращаем только новое сообщение.
    return {"messages": [tool_message]}


# ОБНОВЛЕННЫЙ ГЕНЕРАТОР ОТВЕТА
async def response_generator_node(state: AgentState, llm: ChatOpenAI):
    logger.info("--- Вход в Response Generator ---")
    if TEST_TAVILY_ONLY_MODE:
        logger.warning(
            "!!! РЕЖИМ ТЕСТИРОВАНИЯ TAVILY API: Response Generator не будет вызван. !!!"
        )
        # Просто возвращаем какое-то сообщение, чтобы граф завершился, если он сюда попадет по ошибке.
        return {
            "messages": [
                AIMessage(content="Тест Tavily завершен. Ответ LLM не генерировался.")
            ]
        }

    messages_for_llm = []
    current_messages = list(state.get("messages", []))

    if not current_messages:
        logger.error("Нет сообщений в состоянии для генерации ответа.")
        return {
            "messages": [
                AIMessage(content="Внутренняя ошибка: нет сообщений для обработки.")
            ]
        }

    # Гарантируем, что SystemMessage будет первым и только один
    if not isinstance(current_messages[0], SystemMessage):
        messages_for_llm.append(SystemMessage(content=SYSTEM_PROMPT_RESPONSE_GENERATOR))
        messages_for_llm.extend(current_messages)
    else:
        current_messages[0] = SystemMessage(
            content=SYSTEM_PROMPT_RESPONSE_GENERATOR
        )  # Обновляем, если уже есть
        messages_for_llm.extend(current_messages)

    logger.debug(f"Сообщения, подготовленные для LLM (до вызова): {messages_for_llm}")

    # Валидация последовательности (особенно ToolMessage после AIMessage с tool_calls)
    valid_sequence = True
    for i in range(1, len(messages_for_llm)):  # Начинаем с 1, так как смотрим на i-1
        current_msg = messages_for_llm[i]
        prev_msg = messages_for_llm[i - 1]
        if isinstance(current_msg, ToolMessage):
            if not isinstance(prev_msg, AIMessage) or not prev_msg.tool_calls:
                logger.error(
                    f"Ошибка валидации: ToolMessage (id: {current_msg.tool_call_id}, name: {current_msg.name}) на позиции {i} "
                    f"не предваряется AIMessage с tool_calls. Предшествующее сообщение: {type(prev_msg)}"
                )
                valid_sequence = False
                break
            # Дополнительно проверяем, что tool_call_id из ToolMessage есть в tool_calls предыдущего AIMessage
            found_matching_id = any(
                tc.get("id") == current_msg.tool_call_id for tc in prev_msg.tool_calls
            )
            if not found_matching_id:
                logger.error(
                    f"Ошибка валидации: ToolMessage (id: {current_msg.tool_call_id}, name: {current_msg.name}) "
                    f"не имеет соответствующего ID в tool_calls предыдущего AIMessage: {prev_msg.tool_calls}"
                )
                valid_sequence = False
                break

    if not valid_sequence:
        error_content = "Произошла ошибка при обработке вашего запроса из-за внутренней проблемы с последовательностью вызова инструментов."
        return {"messages": [AIMessage(content=error_content)]}

    try:
        response_ai_message: AIMessage = await llm.ainvoke(messages_for_llm)
        logger.info(
            f"Получено от LLM: {response_ai_message.content if response_ai_message else 'No content'}"
        )
        # Убедимся, что это AIMessage
        if not isinstance(response_ai_message, AIMessage):
            logger.warning(
                f"LLM вернула не AIMessage ({type(response_ai_message)}), оборачиваю."
            )
            response_ai_message = AIMessage(content=str(response_ai_message or ""))
        return {"messages": [response_ai_message]}
    except Exception as e:
        logger.error(
            f"Ошибка при вызове LLM в response_generator_node: {e}", exc_info=True
        )
        return {
            "messages": [AIMessage(content=f"Извините, ошибка генерации ответа: {e}")]
        }


# Проверка вывода (без изменений)
def output_guardrails_node(state: AgentState):
    logger.info("--- Вход в Output Guardrails ---")
    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, AIMessage):
            logger.info(f"Финальный ответ для проверки: {last_message.content}")
    # TODO: Реализовать логику output guardrails (например, проверка на PII, токсичность, галлюцинации)
    # Этап валидации. Если что-то не так, можно изменить сообщение или вернуть ошибку.
    # Пока что просто возвращаем текущие сообщения.
    return {"messages": messages}


# --- Построение графа (ОБНОВЛЕНО) ---
def build_graph(llm: ChatOpenAI):
    """Строит граф LangGraph с умным роутером."""
    workflow = StateGraph(AgentState)

    bound_router_node = functools.partial(router_node, llm=llm)
    bound_response_generator_node = functools.partial(response_generator_node, llm=llm)

    workflow.add_node("input_guardrails", input_guardrails_node)
    workflow.add_node("router", bound_router_node)
    workflow.add_node("tool_executor", tool_executor_node)
    workflow.add_node("tool_output_guardrails", tool_output_guardrails_node)
    workflow.add_node("handle_tool_result", handle_tool_result_node)
    workflow.add_node("generate_response", bound_response_generator_node)
    workflow.add_node("output_guardrails", output_guardrails_node)

    workflow.set_entry_point("input_guardrails")
    workflow.add_edge("input_guardrails", "router")

    workflow.add_conditional_edges(
        "router",
        lambda state: state.get("next_node"),
        {
            "tool_executor": "tool_executor",
            "generate_response": "generate_response",
            # END не добавляем сюда, т.к. роутер не должен напрямую завершать граф.
            # Завершение происходит после output_guardrails.
        },
    )

    workflow.add_edge("tool_executor", "tool_output_guardrails")
    workflow.add_edge("tool_output_guardrails", "handle_tool_result")
    # После обработки результата ИНСТРУМЕНТА, мы снова идем в РОУТЕР.
    # Роутер решит, нужно ли вызывать ЕЩЕ один инструмент или можно генерировать ответ.
    if TEST_TAVILY_ONLY_MODE:
        # В режиме теста Tavily, после обработки результата идем в конец
        logger.warning(
            "!!! РЕЖИМ ТЕСТИРОВАНИЯ TAVILY API: Граф будет завершаться после handle_tool_result -> output_guardrails -> END !!!"
        )
        workflow.add_edge(
            "handle_tool_result", "output_guardrails"
        )  # Пропускаем generate_response и второй router
    else:
        # Нормальный режим: после обработки результата идем снова в роутер
        workflow.add_edge("handle_tool_result", "router")

    workflow.add_edge("generate_response", "output_guardrails")
    workflow.add_edge("output_guardrails", END)

    app = workflow.compile()
    logger.info(
        f"Граф успешно скомпилирован. TEST_TAVILY_ONLY_MODE = {TEST_TAVILY_ONLY_MODE}"
    )
    return app


# --- Инициализация агента (setup_agent) ---
def setup_agent():
    logger.info("Инициализация агента...")

    llm_model_name = os.getenv("LLM_MODEL")
    yc_folder_id = os.getenv("YC_FOLDER_ID")
    yc_api_key = os.getenv("YC_API_KEY")
    # yc_iam_token = os.getenv("YC_IAM_TOKEN") # Альтернатива api_key

    llm = None

    if llm_model_name and llm_model_name.startswith("yandex/") and YANDEX_VERFÜGBAR:
        model_path_segment = llm_model_name.split("/", 1)[
            1
        ]  # yandexgpt/rc -> yandexgpt-rc
        model_uri = (
            f"gpt://{yc_folder_id}/{model_path_segment.replace('/', '-')}/latest"
        )

        if not yc_folder_id:
            logger.error(
                "YC_FOLDER_ID не установлен. Невозможно использовать YandexGPT."
            )
        if not yc_api_key:  # and not yc_iam_token:
            logger.error(
                "YC_API_KEY (или YC_IAM_TOKEN) не установлен. Невозможно использовать YandexGPT."
            )

        if yc_folder_id and yc_api_key:  # или yc_iam_token
            try:
                logger.info(f"Попытка инициализации YandexGPT с model_uri: {model_uri}")
                # Примечание: YandexGPT из langchain_community.llms может ожидать немного другой интерфейс
                # для чат-подобных вызовов, чем ChatOpenAI. Langchain абстрагирует это,
                # но нужно быть внимательным к передаче `messages`.
                # Если есть ChatYandexGPT, он предпочтительнее. Пока используем YandexGPT.
                llm = YandexGPT(
                    api_key=yc_api_key,
                    # iam_token=yc_iam_token, # если используется IAM токен
                    folder_id=yc_folder_id,
                    model_uri=model_uri,
                    temperature=0.1,  # Пример температуры
                    max_tokens=1500,  # Пример лимита токенов
                )
                logger.info("YandexGPT успешно инициализирован.")
            except Exception as e:
                logger.error(f"Ошибка при инициализации YandexGPT: {e}", exc_info=True)
                llm = None  # Возврат к OpenAI, если YandexGPT не удалось настроить
        else:
            logger.warning(
                "Недостаточно данных для инициализации YandexGPT (нужен YC_FOLDER_ID и YC_API_KEY/YC_IAM_TOKEN)."
            )
            llm = None

    if llm is None:  # Если YandexGPT не был настроен или выбран OpenAI
        openai_model_to_use = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
        logger.info(f"Используется модель OpenAI: {openai_model_to_use}")
        openai_api_base_proxy = os.getenv("OPENAI_API_BASE_PROXY")
        openai_api_key_env = os.getenv("OPENAI_API_KEY")

        if not openai_api_key_env:
            logger.error(
                "OPENAI_API_KEY не найден в переменных окружения. OpenAI не будет работать."
            )
            # В этом случае агент не сможет работать, если YandexGPT тоже не настроен.
            # Можно либо бросить исключение, либо он просто не будет отвечать.
            # Для учебных целей оставим так, но в проде нужно обработать.
            raise ValueError(
                "Не удалось настроить ни одну LLM. Проверьте переменные окружения."
            )

        if openai_api_base_proxy:
            logger.info(f"Используется OpenAI через прокси: {openai_api_base_proxy}")
            llm = ChatOpenAI(
                model=openai_model_to_use,
                temperature=0,
                openai_api_base=openai_api_base_proxy,
                openai_api_key=openai_api_key_env,
            )
        else:
            logger.warning(
                "OPENAI_API_BASE_PROXY не найден. Используется стандартный URL OpenAI."
            )
            llm = ChatOpenAI(
                model=openai_model_to_use,
                temperature=0,
                openai_api_key=openai_api_key_env,
            )
        logger.info("ChatOpenAI успешно инициализирован.")

    tool_registry.clear()
    tools_to_register = [
        WebSearchTool(),
        SearchFAQTool(),
        AddFAQTool(),
        UpdateFAQTool(),
        DeleteFAQTool(),
        TavilyYandexCloudSearchTool(),
    ]

    for tool_instance in tools_to_register:
        try:
            tool_registry[tool_instance.name] = tool_instance
            logger.info(
                f"Инструмент '{tool_instance.name}' успешно инициализирован и зарегистрирован."
            )
        except Exception as e:
            logger.error(
                f"Ошибка при инициализации или регистрации инструмента {getattr(tool_instance, 'name', 'UNKNOWN')}: {e}",
                exc_info=True,
            )

    if (
        not tool_registry.get(TavilyYandexCloudSearchTool().name)
        and TEST_TAVILY_ONLY_MODE
    ):
        logger.error(
            f"КРИТИЧЕСКАЯ ОШИБКА: Инструмент '{TavilyYandexCloudSearchTool().name}' не зарегистрирован, но включен режим теста Tavily! Тест не будет работать корректно."
        )
    elif not tool_registry and not TEST_TAVILY_ONLY_MODE:
        logger.warning(
            "Внимание: Ни один инструмент не был успешно инициализирован! Агент будет работать без инструментов (кроме режима теста Tavily)."
        )
    elif not TEST_TAVILY_ONLY_MODE:
        logger.info(f"Зарегистрированные инструменты: {list(tool_registry.keys())}")

    app = build_graph(llm)
    logger.info("Агент успешно настроен и граф скомпилирован.")
    return app
