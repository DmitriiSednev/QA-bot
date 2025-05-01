import operator
import os
import functools
import logging # Добавляем для логгирования ключей
from typing import Sequence, Literal, Type, Dict, Any, Optional, List, Union, Generator, Callable

# --- Настройка логгера для этого модуля ---
logger = logging.getLogger(__name__)

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    AIMessage,
    HumanMessage,
    ToolMessage,
)
# Возвращаем ChatOpenAI, так как используется OpenAI-совместимый прокси
from langchain_openai import ChatOpenAI
# from langchain_yandex import ChatYandexGPT # Убираем YandexGPT
from langgraph.graph import END, StateGraph
from langchain.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool # Убедимся, что BaseTool импортирован
from langchain.agents import create_openai_functions_agent
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langgraph.graph import Graph
from langgraph.prebuilt import ToolNode

from .state import AgentState

# --- Импорт инструментов ---
from .tools import (
    SearchFAQTool,
    AddFAQTool,
    UpdateFAQTool,
    DeleteFAQTool,
    ContextAnalyzerTool,
    YandexCloudSearchTool,
    ChatHistorySearchTool
)

# TODO: Импортировать другие инструменты (update_faq, delete_faq и т.д.)

# --- Определяем "инструменты" ---
# Описания для LLM-роутера
tool_descriptions: Dict[str, str] = {
    "search_faq": "Искать ответ на вопрос пользователя ВНУТРИ базы знаний FAQ (ПЕРВЫЙ ИСТОЧНИК). Использовать для специфичных знаний проекта.",
    "add_faq": "Добавить новую пару вопрос-ответ в базу знаний FAQ.",
    "update_faq": "Изменить существующую запись в FAQ по её ID.",
    "delete_faq": "Удалить запись из FAQ по её ID.",
    "yandex_cloud_docs_search": "Искать информацию ИСКЛЮЧИТЕЛЬНО в официальной документации Yandex Cloud (yandex.cloud/ru/docs/). Использовать ТОЛЬКО для вопросов о Yandex Cloud, и ТОЛЬКО ЕСЛИ поиск по FAQ ('search_faq') не дал ответа (ВТОРОЙ ИСТОЧНИК).",
    "search_chat_history": "Искать похожие вопросы и ответы в истории чата. Использовать, когда нужно найти ранее заданные похожие вопросы.",
    "context_analyzer": "Анализировать контекст разговора и извлекать ключевые темы и вопросы."
}

# Создаем экземпляры инструментов
tools = [
    SearchFAQTool(),
    AddFAQTool(),
    UpdateFAQTool(),
    DeleteFAQTool(),
    YandexCloudSearchTool(),
    ChatHistorySearchTool(),
    ContextAnalyzerTool()
]

# Реестр реальных объектов инструментов
# Используем классы напрямую для создания экземпляров
tool_classes: List[Type[BaseTool]] = [
    SearchFAQTool,
    AddFAQTool,
    UpdateFAQTool,
    DeleteFAQTool,
    YandexCloudSearchTool,
    ChatHistorySearchTool,
    ContextAnalyzerTool,
]

# Создаем экземпляры инструментов (можно передавать параметры конфигурации сюда, если нужно)
tool_registry: Dict[str, BaseTool] = {tool_cls().name: tool_cls() for tool_cls in tool_classes}

# Убедимся, что имена совпадают с ключами в tool_descriptions
for name in tool_descriptions:
    if name not in tool_registry:
        logger.warning(f"Описание для инструмента '{name}' есть, но сам инструмент не найден в реестре.")
for name in tool_registry:
    if name not in tool_descriptions:
        logger.warning(f"Инструмент '{name}' есть в реестре, но его описания нет в tool_descriptions.")

# COMMAND_TOOLS = {"add_faq", "update_faq", "delete_faq"}

# Реализация ToolExecutor
class ToolExecutor:
    """Класс для выполнения инструментов."""
    
    def __init__(self, tools: List[BaseTool]):
        self.tools = {tool.name: tool for tool in tools}
    
    def execute(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
        """Выполняет инструмент с заданными параметрами."""
        if tool_name not in self.tools:
            raise ValueError(f"Инструмент {tool_name} не найден")
        return self.tools[tool_name].invoke(tool_input)

def input_guardrails_node(state: AgentState) -> Dict[str, Any]:
    """Проверяет входные данные."""
    logger.info("--- Вход в Input Guardrails ---")
    try:
        # TODO: Реализовать логику проверки ввода
        return {}
    except Exception as e:
        logger.error(f"Ошибка в input_guardrails_node: {e}", exc_info=True)
        return {"error": str(e)}


# ---- НОВЫЙ УМНЫЙ РОУТЕР ----
def router_node(state: AgentState, llm: ChatOpenAI) -> Dict[str, Any]:
    """Анализирует сообщение пользователя и решает, какой инструмент вызвать или генерировать ответ."""
    logger.info("--- Вход в Умный Роутер ---")
    try:
        messages = state["messages"]
        last_message = messages[-1]
        logger.info(f"Последнее сообщение для роутера: {last_message.content}")

        # Привязываем инструменты к LLM
        llm_with_tools = llm.bind_tools(list(tool_registry.values()))

        if isinstance(last_message, ToolMessage):
            logger.info("Последнее сообщение - результат инструмента. Переход к генерации ответа.")
            return {"next_node": "generate_response"}

        logger.info("Запрос к LLM-роутеру с привязанными инструментами...")
        ai_message = llm_with_tools.invoke(messages)
        logger.info(f"Ответ LLM-роутера: {ai_message}")

        if not hasattr(ai_message, "tool_calls") or not ai_message.tool_calls:
            logger.info("Router node: LLM не выбрала инструмент.")
            return {
                "next_node": "generate_response",
                "messages": state["messages"] + [ai_message],
            }

        tool_call = ai_message.tool_calls[0]
        tool_name = tool_call["name"]
        tool_input = tool_call["args"]
        tool_call_id = tool_call.get("id")

        logger.info(f"LLM выбрала инструмент: '{tool_name}' с аргументами: {tool_input}")
        logger.info(f"Router node: Извлеченный tool_call_id: {tool_call_id}")

        return_state = {
            "next_node": "tool_executor",
            "tool_to_call": tool_name,
            "tool_input": tool_input,
            "current_tool_call_id": tool_call_id,
            "messages": [ai_message],
        }
        logger.info(f"Router node: Возвращаемое состояние: {return_state}")
        return return_state

    except Exception as e:
        logger.error(f"Ошибка в LLM-роутере: {e}", exc_info=True)
        error_message = AIMessage(content=f"Ошибка роутера: {e}")
        return {
            "next_node": "generate_response",
            "messages": state["messages"] + [error_message],
        }


# Исполнитель инструментов (добавляем возврат имени инструмента И current_tool_call_id)
def tool_executor_node(state: AgentState) -> Dict[str, Any]:
    """Вызывает выбранный роутером инструмент и передает current_tool_call_id дальше."""
    logger.info("--- Вход в Tool Executor ---")
    try:
        tool_name = state.get("tool_to_call")
        tool_input = state.get("tool_input", {})
        current_tool_call_id = state.get("current_tool_call_id")
        logger.info(
            f"Вызов инструмента: {tool_name} с вводом: {tool_input}, ID вызова: {current_tool_call_id}"
        )

        if not tool_name:
            logger.error("Ошибка: Не указано имя инструмента для вызова.")
            return {
                "tool_result": "Ошибка: Не указан инструмент.",
                "tool_name_executed": None,
                "current_tool_call_id": current_tool_call_id,
            }

        tool_to_execute = tool_registry.get(tool_name)
        if not tool_to_execute:
            logger.error(f"Ошибка: Инструмент '{tool_name}' не найден в реестре.")
            return {
                "tool_result": f"Ошибка: Инструмент '{tool_name}' не найден.",
                "tool_name_executed": tool_name,
                "current_tool_call_id": current_tool_call_id,
            }

        # Прямой вызов инструмента с распаковкой аргументов
        result = tool_to_execute.invoke(tool_input)
        logger.info(f"Результат инструмента '{tool_name}': {result}")
        
        return {
            "tool_result": result,
            "tool_name_executed": tool_name,
            "current_tool_call_id": current_tool_call_id,
        }
    except Exception as e:
        logger.error(f"Ошибка при выполнении инструмента: {e}", exc_info=True)
        return {
            "tool_result": f"Ошибка при выполнении инструмента: {e}",
            "tool_name_executed": tool_name if 'tool_name' in locals() else None,
            "current_tool_call_id": current_tool_call_id,
        }


# Проверка вывода инструмента (ИЗМЕНЕНО: передает нужные поля дальше)
def tool_output_guardrails_node(state: AgentState):
    """(Заглушка) Проверяет вывод инструмента и передает ключевые поля дальше."""
    logger.info("--- Вход в Tool Output Guardrails ---")
    tool_result = state.get("tool_result")
    tool_name_executed = state.get("tool_name_executed")
    current_tool_call_id = state.get("current_tool_call_id")
    logger.info(
        f"Результат инструмента '{tool_name_executed}' (ID: {current_tool_call_id}) для проверки: {tool_result}"
    )
    # TODO: Реализовать логику проверки вывода

    # Явно передаем нужные поля дальше, чтобы они не потерялись
    return {
        "tool_result": tool_result,
        "tool_name_executed": tool_name_executed,
        "current_tool_call_id": current_tool_call_id,
    }


# ---- НОВЫЙ УЗЕЛ для обработки результата инструмента ----
def handle_tool_result_node(state: AgentState):
    """Преобразует результат инструмента в ToolMessage и добавляет к истории,
    используя правильный tool_call_id из последнего AIMessage.
    """
    logger.info("--- Вход в Handle Tool Result --- ")
    logger.info(f"Handle Tool Result: Входящее состояние: {state}")
    tool_result = state.get("tool_result")
    tool_name_executed = state.get("tool_name_executed")
    # tool_to_call = state.get("tool_to_call") # Имя инструмента, который *запросили* (уже не так важно здесь)

    if not tool_name_executed:
        logger.warning(
            "Не удалось определить имя выполненного инструмента для обработки результата."
        )
        # Просто возвращаем текущие сообщения, чтобы не прерывать поток
        return {"messages": state["messages"]}

    # --- Ищем последний AIMessage с tool_calls, чтобы извлечь ID ---
    last_ai_message_with_tool_calls = None
    extracted_tool_call_id = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            last_ai_message_with_tool_calls = msg
            # Предполагаем, что роутер вызывает только один инструмент за раз
            if msg.tool_calls:
                extracted_tool_call_id = msg.tool_calls[0]["id"]
            break  # Нашли последнее нужное сообщение, выходим

    logger.info(f"Handle Tool Result: Извлеченный tool_call_id: {extracted_tool_call_id}")

    # --- Создаем ToolMessage ---
    if not extracted_tool_call_id:
        # Если не нашли ID (очень странно), используем старый fallback
        logger.warning(
            f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось извлечь tool_call_id из AIMessage для инструмента {tool_name_executed}! Использую fallback."
        )
        tool_message = ToolMessage(
            content=f"Критическая ошибка: не найден tool_call_id для {tool_name_executed}. Результат: {tool_result}",
            name=tool_name_executed,  # Используем имя реально выполненного инструмента
            tool_call_id="error_no_tool_call_id_found",
        )
    else:
        # Нормальный случай: создаем ToolMessage с реальным ID
        tool_message = ToolMessage(
            content=str(tool_result),
            name=tool_name_executed,
            tool_call_id=extracted_tool_call_id,
        )
        logger.info(f"Создано ToolMessage: {tool_message}")

    # Добавляем ToolMessage к истории сообщений
    return {"messages": state["messages"] + [tool_message]}


# ОБНОВЛЕННЫЙ ГЕНЕРАТОР ОТВЕТА: теперь основной источник контекста - история сообщений
def response_generator_node(state: AgentState, llm):
    """Генерирует финальный ответ, основываясь на всей истории сообщений."""
    logger.info("--- Вход в Response Generator ---")
    messages_for_llm = state["messages"]
    logger.info(f"Сообщения для генерации: {messages_for_llm}")

    # Контекст теперь полностью в messages_for_llm (включая ToolMessage)
    # Можно удалить старую логику добавления faq_result/tool_result как AIMessage

    # Обновляем системный промпт с указанием приоритета источников
    system_prompt = """Ты - полезный AI ассистент, отвечающий на вопросы пользователей.

Приоритет источников информации:
1.  Сначала используй результаты поиска по внутренней базе знаний FAQ (если был вызван инструмент 'search_faq'). Это самый достоверный источник для специфичных вопросов.
2.  Если поиск по FAQ не дал ответа И вопрос касается Yandex Cloud, используй результаты поиска по официальной документации Yandex Cloud (если был вызван инструмент 'yandex_cloud_docs_search'). Помни, что доступ к документации может быть ограничен.
3.  Не используй общие знания или поиск в интернете, если это не предусмотрено специальными инструментами (которых сейчас нет, кроме поиска по документации YC).

Отвечай на последний вопрос пользователя ясно и по делу, учитывая всю предыдущую историю диалога и результаты вызова инструментов (ToolMessage).
Основывай свой ответ на результатах инструментов, если они релевантны и доступны.
Если результат поиска по документации недоступен или нерелевантен, так и скажи.
Если последний запрос был на выполнение действия (add/update/delete) и он выполнен успешно (видно из ToolMessage), просто подтверди это кратко.
Если пользователь спрашивает о твоем происхождении, создателях или о том, кто тебя сделал (независимо от формулировки), отвечай только: "Меня сделали в "YandexGPT"".
"""
    # Проверяем, есть ли системное сообщение, и добавляем/заменяем его
    if not messages_for_llm or not isinstance(messages_for_llm[0], SystemMessage):
        messages_for_llm = [SystemMessage(content=system_prompt)] + messages_for_llm
    elif messages_for_llm[0].content != system_prompt:  # Обновляем, если изменился
        messages_for_llm[0] = SystemMessage(content=system_prompt)

    logger.info(f"Отправка в LLM: {messages_for_llm}")
    try:
        response_message = llm.invoke(messages_for_llm)
        logger.info(f"Получено от LLM: {response_message}")

        if not isinstance(response_message, AIMessage):
            # Преобразуем, если LLM вернула строку или что-то иное
            response_message = AIMessage(content=str(response_message))

        logger.info(f"Ответ LLM: {response_message.content}")
        # Возвращаем ТОЛЬКО новое сообщение AI, оно добавится к state['messages'] оператором '+' или operator.add
        return {"messages": [response_message]}
    except Exception as e:
        logger.error(f"Ошибка при вызове LLM: {e}")
        error_message = AIMessage(
            content=f"Извините, произошла ошибка при генерации ответа: {e}"
        )
        return {"messages": [error_message]}


# Проверка вывода (без изменений)
def output_guardrails_node(state: AgentState):
    """(Заглушка) Проверяет финальный ответ.
    В будущем здесь будет логика Output Guardrails.
    """
    logger.info("--- Вход в Output Guardrails ---")
    # TODO: Реализовать логику проверки вывода
    # print(f"Финальный ответ: {state['messages'][-1].content}")
    return {}


# --- Построение графа (ОБНОВЛЕНО) ---


def build_graph(llm: ChatOpenAI) -> StateGraph:
    """Создает и компилирует граф LangGraph для QA-бота."""
    try:
        # Инициализация графа
        workflow = StateGraph(AgentState)
        
        # Добавляем узлы
        workflow.add_node("input_guardrails", input_guardrails_node)
        workflow.add_node("analyze_context", analyze_context)
        workflow.add_node("router", router_node)
        workflow.add_node("tool_executor", tool_executor_node)
        workflow.add_node("tool_output_guardrails", tool_output_guardrails_node)
        workflow.add_node("handle_tool_result", handle_tool_result_node)
        workflow.add_node("response_generator", response_generator_node)
        workflow.add_node("output_guardrails", output_guardrails_node)
        
        # Добавляем ребра
        workflow.add_conditional_edges(
            "analyze_context",
            should_respond,
            {
                "agent": "router",
                "end": END
            }
        )
        
        workflow.add_edge("input_guardrails", "analyze_context")
        workflow.add_edge("router", "tool_executor")
        workflow.add_edge("tool_executor", "tool_output_guardrails")
        workflow.add_edge("tool_output_guardrails", "handle_tool_result")
        workflow.add_edge("handle_tool_result", "response_generator")
        workflow.add_edge("response_generator", "output_guardrails")
        workflow.add_edge("output_guardrails", END)
        
        # Устанавливаем начальный узел
        workflow.set_entry_point("input_guardrails")
        
        return workflow.compile()
        
    except Exception as e:
        logger.error(f"Ошибка при создании графа: {e}", exc_info=True)
        raise


# --- Инициализация агента (setup_agent) ---


def create_agent_workflow(tools: list[BaseTool]):
    """Создает рабочий процесс агента с инструментами."""
    # TODO: Реализовать создание рабочего процесса
    pass

def setup_agent() -> Optional[dict]:
    """Настраивает агента с необходимыми инструментами."""
    try:
        # Инициализация инструментов
        tools = [
            SearchFAQTool(),
            AddFAQTool(),
            UpdateFAQTool(),
            DeleteFAQTool(),
            YandexCloudSearchTool(),
            ChatHistorySearchTool(),
            ContextAnalyzerTool()
        ]
        
        # Создание графа
        workflow = create_agent_workflow(tools)
        
        return workflow
    except Exception as e:
        logger.error(f"Ошибка при настройке агента: {e}")
        return None


# --- Основная функция вызова агента ---
# (остается без изменений, т.к. работает с скомпилированным app)

# --- Запуск, если файл выполняется напрямую (для тестов) ---
if __name__ == "__main__":
    # Настройка логирования для теста
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Запуск agent_executor как основного скрипта для теста.")

    # Загрузка переменных окружения (нужен python-dotenv)
    try:
        from dotenv import load_dotenv
        load_dotenv()
        logger.info(".env файл загружен.")
    except ImportError:
        logger.warning("python-dotenv не установлен. Переменные окружения должны быть установлены вручную.")

    # Настройка и получение скомпилированного агента
    compiled_agent = setup_agent()

    # Пример вызова агента (имитация ввода пользователя)
    # Убедитесь, что у вас есть .env файл с API_KEY и др.
    if os.getenv("API_KEY") and os.getenv("API_BASE"):
        print("\n--- Тестовый вызов агента (через прокси) ---")
        config = {"configurable": {"thread_id": "test-thread-1"}}
        user_input = "Что такое Yandex Managed Service for Kubernetes?"

        # Используем stream для получения событий
        events = compiled_agent.stream(
            {"messages": [HumanMessage(content=user_input)]}, config=config
        )

        print(f"Ввод пользователя: {user_input}")
        print("Ответ агента:")
        final_response = None
        for event in events:
            # Печатаем события для отладки
            print(f"Event: {event}")
            # Ищем финальный ответ в событии 'response_generator'
            # (Предполагая, что узел называется 'response_generator' и возвращает AIMessage в 'messages')
            if event.get("event") == "on_chain_end" and event.get("name") == "response_generator":
                 # Получаем последнее сообщение из состояния на выходе узла
                 output_messages = event.get("data", {}).get("output", {}).get("messages", [])
                 if output_messages and isinstance(output_messages[-1], AIMessage):
                     final_response = output_messages[-1].content

        print("\n--- Финальный ответ ---")
        if final_response:
            print(final_response)
        else:
            print("Не удалось получить финальный ответ от агента.")
        print("--- Тестовый вызов завершен ---")
    else:
        print("Пропуск тестового вызова: API_KEY или API_BASE для прокси не установлены.")

def execute_tools(state: AgentState) -> Dict[str, Any]:
    """Выполняет инструменты, выбранные агентом."""
    try:
        tool_calls = state.get("tool_calls", [])
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            
            if tool_name in tool_registry:
                tool = tool_registry[tool_name]
                result = tool.invoke(tool_args)
                results.append(result)
            else:
                logger.warning(f"Инструмент {tool_name} не найден")
                
        return {"tool_results": results}
    except Exception as e:
        logger.error(f"Ошибка при выполнении инструментов: {e}", exc_info=True)
        return {"error": str(e)}

def analyze_context(state: AgentState) -> Dict[str, Any]:
    """Анализирует контекст сообщения."""
    logger.info("--- Анализ контекста ---")
    try:
        context = state.get("context", {})
        if not context:
            return {"should_respond": True, "confidence": 1.0}

        chat_type = context.get("chat_type", "private")
        is_mentioned = context.get("is_mentioned", False)
        is_reply_to_bot = context.get("is_reply_to_bot", False)
        
        # В личных сообщениях всегда отвечаем
        if chat_type == "private":
            return {"should_respond": True, "confidence": 1.0}
            
        # Если упомянут или ответ на бота - отвечаем
        if is_mentioned or is_reply_to_bot:
            return {"should_respond": True, "confidence": 0.9}
            
        return {"should_respond": False, "confidence": 0.0}
    except Exception as e:
        logger.error(f"Ошибка в analyze_context: {e}", exc_info=True)
        return {"error": str(e)}

def should_respond(state: AgentState) -> str:
    """Определяет следующий узел на основе анализа контекста."""
    try:
        should_respond = state.get("should_respond", False)
        if should_respond:
            return "agent"
        return "end"
    except Exception as e:
        logger.error(f"Ошибка в should_respond: {e}", exc_info=True)
        return "end"
