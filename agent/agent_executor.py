import operator
import os
import functools
from typing import Sequence, Literal

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langchain.prompts import ChatPromptTemplate

from .state import AgentState

# --- Импорт инструментов ---
from .tools.web_search import WebSearchTool
from .tools.search_faq import SearchFAQTool, SearchFAQInput
from .tools.add_faq import AddFAQTool
from .tools.update_faq import UpdateFAQTool
from .tools.delete_faq import DeleteFAQTool

# TODO: Импортировать другие инструменты (update_faq, delete_faq и т.д.)

# --- Определяем "инструменты" ---
# Описания для LLM-роутера
tool_descriptions = {
    "search_faq": "Искать ответ на вопрос пользователя в базе знаний FAQ (предпочтительно для специфичных знаний проекта).",
    "add_faq": "Добавить новую пару вопрос-ответ в базу знаний FAQ.",
    "update_faq": "Изменить существующую запись в FAQ по её ID.",
    "delete_faq": "Удалить запись из FAQ по её ID.",
    "web_search": "Искать актуальную информацию в интернете (использовать для общих знаний, фактов, новостей, когда FAQ не помог).",
}

# Реестр реальных объектов инструментов
tool_registry = {}

# COMMAND_TOOLS = {"add_faq", "update_faq", "delete_faq"}


def input_guardrails_node(state: AgentState):
    """(Заглушка) Проверяет входные данные.
    В будущем здесь будет логика Input Guardrails.
    """
    print("--- Вход в Input Guardrails ---")
    # TODO: Реализовать логику проверки ввода
    return {}


# ---- НОВЫЙ УМНЫЙ РОУТЕР ----
def router_node(state: AgentState, llm):
    """Анализирует сообщение пользователя и решает, какой инструмент вызвать или генерировать ответ."""
    print("--- Вход в Умный Роутер ---")
    messages = state["messages"]
    last_message = messages[-1]
    print(f"Последнее сообщение для роутера: {last_message.content}")

    # Привязываем инструменты к LLM
    llm_with_tools = llm.bind_tools(list(tool_registry.values()))

    # Формируем промпт для роутера (можно сделать более сложным при необходимости)
    # Пока просто передаем историю сообщений
    # Важно: Если последний ToolMessage, LLM должна генерировать ответ, а не вызывать инструмент снова.
    # Добавим проверку на это.
    if isinstance(last_message, ToolMessage):
        print(
            "Последнее сообщение - результат инструмента. Переход к генерации ответа."
        )
        return {"next_node": "generate_response"}

    print("Запрос к LLM-роутеру с привязанными инструментами...")
    try:
        # Вызываем LLM, которая может решить вызвать инструмент
        ai_message = llm_with_tools.invoke(messages)
        print(f"Ответ LLM-роутера: {ai_message}")

        if not hasattr(ai_message, "tool_calls") or not ai_message.tool_calls:
            # Если LLM не вызвала инструмент, генерируем ответ напрямую
            print("Router node: LLM не выбрала инструмент.")  # Отладка
            return {
                "next_node": "generate_response",
                "messages": state["messages"] + [ai_message],
            }

        # Если LLM вызвала инструмент
        # Берем ПЕРВЫЙ вызов инструмента (для простоты)
        tool_call = ai_message.tool_calls[0]
        tool_name = tool_call["name"]
        tool_input = tool_call["args"]
        tool_call_id = tool_call.get("id")

        print(f"LLM выбрала инструмент: '{tool_name}' с аргументами: {tool_input}")
        print(f"Router node: Извлеченный tool_call_id: {tool_call_id}")

        # Добавляем сообщение AI с запросом на вызов инструмента в историю
        # УДАЛЯЕМ ручное добавление: state["messages"] += [ai_message]

        return_state = {
            "next_node": "tool_executor",
            "tool_to_call": tool_name,
            "tool_input": tool_input,
            "current_tool_call_id": tool_call_id,
            "messages": [ai_message],  # ВОЗВРАЩАЕМ сообщение AI здесь
        }
        print(f"Router node: Возвращаемое состояние: {return_state}")
        return return_state

    except Exception as e:
        print(f"Ошибка в LLM-роутере: {e}. Переходим к генерации.")
        # Возвращаем ошибку в сообщении?
        error_message = AIMessage(content=f"Ошибка роутера: {e}")
        # state['messages'] += [error_message] # Добавлять ли ошибку?
        return {
            "next_node": "generate_response",
            "messages": state["messages"] + [error_message],
        }


# Исполнитель инструментов (добавляем возврат имени инструмента И current_tool_call_id)
def tool_executor_node(state: AgentState):
    """Вызывает выбранный роутером инструмент и передает current_tool_call_id дальше."""
    print("--- Вход в Tool Executor ---")
    tool_name = state.get("tool_to_call")
    tool_input = state.get("tool_input")
    current_tool_call_id = state.get("current_tool_call_id")
    print(
        f"Вызов инструмента: {tool_name} с вводом: {tool_input}, ID вызова: {current_tool_call_id}"
    )

    if not tool_name:
        print("Ошибка: Не указано имя инструмента для вызова.")
        # Возвращаем ошибку, имя (None) и ID (None)
        return {
            "tool_result": "Ошибка: Не указан инструмент.",
            "tool_name_executed": None,
            "current_tool_call_id": current_tool_call_id,
        }

    tool_to_execute = tool_registry.get(tool_name)
    if not tool_to_execute:
        print(f"Ошибка: Инструмент '{tool_name}' не найден в реестре.")
        return {
            "tool_result": f"Ошибка: Инструмент '{tool_name}' не найден.",
            "tool_name_executed": tool_name,
            "current_tool_call_id": current_tool_call_id,
        }

    if not tool_input:
        print(
            f"Предупреждение: Нет входных данных для инструмента '{tool_name}'. Пробуем без них."
        )
        tool_input = {}

    try:
        result = tool_to_execute.invoke(tool_input)
        print(f"Результат инструмента '{tool_name}': {result}")
        # Возвращаем результат, имя выполненного инструмента И ID вызова
        return {
            "tool_result": result,
            "tool_name_executed": tool_name,
            "current_tool_call_id": current_tool_call_id,
        }
    except Exception as e:
        print(f"Ошибка при выполнении инструмента '{tool_name}': {e}")
        return {
            "tool_result": f"Ошибка при выполнении инструмента '{tool_name}': {e}",
            "tool_name_executed": tool_name,
            "current_tool_call_id": current_tool_call_id,
        }


# Проверка вывода инструмента (ИЗМЕНЕНО: передает нужные поля дальше)
def tool_output_guardrails_node(state: AgentState):
    """(Заглушка) Проверяет вывод инструмента и передает ключевые поля дальше."""
    print("--- Вход в Tool Output Guardrails ---")
    tool_result = state.get("tool_result")
    tool_name_executed = state.get("tool_name_executed")
    current_tool_call_id = state.get("current_tool_call_id")
    print(
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
    print("--- Вход в Handle Tool Result --- ")
    print(f"Handle Tool Result: Входящее состояние: {state}")
    tool_result = state.get("tool_result")
    tool_name_executed = state.get("tool_name_executed")
    # tool_to_call = state.get("tool_to_call") # Имя инструмента, который *запросили* (уже не так важно здесь)

    if not tool_name_executed:
        print(
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

    print(f"Handle Tool Result: Извлеченный tool_call_id: {extracted_tool_call_id}")

    # --- Создаем ToolMessage ---
    if not extracted_tool_call_id:
        # Если не нашли ID (очень странно), используем старый fallback
        print(
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
        print(f"Создано ToolMessage: {tool_message}")

    # Добавляем ToolMessage к истории сообщений
    return {"messages": state["messages"] + [tool_message]}


# ОБНОВЛЕННЫЙ ГЕНЕРАТОР ОТВЕТА: теперь основной источник контекста - история сообщений
def response_generator_node(state: AgentState, llm):
    """Генерирует финальный ответ, основываясь на всей истории сообщений (включая ToolMessage)."""
    print("--- Вход в Response Generator ---")
    messages_for_llm = state["messages"]
    print(f"Сообщения для генерации: {messages_for_llm}")

    # Контекст теперь полностью в messages_for_llm (включая ToolMessage)
    # Можно удалить старую логику добавления faq_result/tool_result как AIMessage

    # Системный промпт (можно упростить, т.к. контекст уже в сообщениях)
    # Или оставить строгим, чтобы он правильно интерпретировал ToolMessage
    system_prompt = """Ты - полезный AI ассистент. Отвечай на последний вопрос пользователя ясно и по делу,
учитывая всю предыдущую историю диалога, включая результаты вызова инструментов (ToolMessage). 
Основывай свой ответ на результатах инструментов, если они релевантны.
Если последний запрос был на выполнение действия (add/update/delete) и он выполнен успешно (видно из ToolMessage), 
просто подтверди это кратко.
"""
    # Проверяем, есть ли системное сообщение, и добавляем/заменяем его
    if not messages_for_llm or not isinstance(messages_for_llm[0], SystemMessage):
        messages_for_llm = [SystemMessage(content=system_prompt)] + messages_for_llm
    elif messages_for_llm[0].content != system_prompt:  # Обновляем, если изменился
        messages_for_llm[0] = SystemMessage(content=system_prompt)

    print(f"Отправка в LLM: {messages_for_llm}")
    try:
        response_message = llm.invoke(messages_for_llm)
        print(f"Получено от LLM: {response_message}")

        if not isinstance(response_message, AIMessage):
            # Преобразуем, если LLM вернула строку или что-то иное
            response_message = AIMessage(content=str(response_message))

        print(f"Ответ LLM: {response_message.content}")
        # Возвращаем ТОЛЬКО новое сообщение AI, оно добавится к state['messages'] оператором '+' или operator.add
        return {"messages": [response_message]}
    except Exception as e:
        print(f"Ошибка при вызове LLM: {e}")
        error_message = AIMessage(
            content=f"Извините, произошла ошибка при генерации ответа: {e}"
        )
        return {"messages": [error_message]}


# Проверка вывода (без изменений)
def output_guardrails_node(state: AgentState):
    """(Заглушка) Проверяет финальный ответ.
    В будущем здесь будет логика Output Guardrails.
    """
    print("--- Вход в Output Guardrails ---")
    # TODO: Реализовать логику проверки вывода
    # print(f"Финальный ответ: {state['messages'][-1].content}")
    return {}


# --- Построение графа (ОБНОВЛЕНО) ---


def build_graph(llm):
    """Строит граф LangGraph с умным роутером."""
    workflow = StateGraph(AgentState)

    # Привязываем LLM к узлам где она нужна
    bound_router_node = functools.partial(router_node, llm=llm)
    bound_response_generator_node = functools.partial(response_generator_node, llm=llm)

    # Добавляем узлы (включая новый handle_tool_result)
    workflow.add_node("input_guardrails", input_guardrails_node)
    # workflow.add_node("search_faq", search_faq_node) # Узел search_faq теперь вызывается через роутер как инструмент
    workflow.add_node("router", bound_router_node)
    workflow.add_node("tool_executor", tool_executor_node)
    workflow.add_node("tool_output_guardrails", tool_output_guardrails_node)
    workflow.add_node("handle_tool_result", handle_tool_result_node)  # Новый узел
    workflow.add_node("generate_response", bound_response_generator_node)
    workflow.add_node("output_guardrails", output_guardrails_node)

    # Точка входа
    workflow.set_entry_point("input_guardrails")

    # Ребра
    workflow.add_edge("input_guardrails", "router")  # После входа идем в роутер

    # Условный переход от роутера
    workflow.add_conditional_edges(
        "router",
        lambda state: state.get("next_node"),  # Роутер сам определяет следующий узел
        {
            "tool_executor": "tool_executor",  # Если роутер выбрал инструмент
            "generate_response": "generate_response",  # Если роутер решил генерировать ответ
        },
    )

    # Путь выполнения инструмента
    workflow.add_edge("tool_executor", "tool_output_guardrails")
    workflow.add_edge(
        "tool_output_guardrails", "handle_tool_result"
    )  # Обрабатываем результат
    workflow.add_edge(
        "handle_tool_result", "generate_response"
    )  # Генерируем ответ ПОСЛЕ обработки результата

    # Финальные шаги
    workflow.add_edge("generate_response", "output_guardrails")
    workflow.add_edge("output_guardrails", END)

    # Компилируем граф
    app = workflow.compile()
    print("Граф успешно скомпилирован с новой логикой (умный роутер).")
    return app


# --- Инициализация агента (setup_agent) ---


def setup_agent():
    """Собирает и компилирует граф агента, инициализируя LLM и инструменты."""

    # Получаем API ключ и базовый URL из переменных окружения
    # .env должен быть уже загружен в main.py
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")

    if not api_key:
        raise ValueError("OPENAI_API_KEY не найден в переменных окружения.")
    if not base_url:
        raise ValueError("OPENAI_API_BASE не найден в переменных окружения.")

    # Инициализируем LLM (используя langchain_openai для совместимости)
    # Добавляем таймаут, чтобы избежать зависаний
    llm = ChatOpenAI(
        api_key=api_key, base_url=base_url, request_timeout=60  # Таймаут в 60 секунд
    )
    # TODO: Добавить обработку ошибок инициализации LLM?

    # --- Инициализация Инструментов ---
    tool_registry.clear()

    # Web Search Tool
    try:
        web_search_tool = WebSearchTool()
        tool_registry[web_search_tool.name] = web_search_tool
        print(f"Инструмент '{web_search_tool.name}' инициализирован.")
    except Exception as e:
        print(f"Ошибка при инициализации WebSearchTool: {e}")

    # Search FAQ Tool
    try:
        search_faq_tool = SearchFAQTool()
        tool_registry[search_faq_tool.name] = search_faq_tool
        print(f"Инструмент '{search_faq_tool.name}' инициализирован.")
    except Exception as e:
        print(f"Ошибка при инициализации SearchFAQTool: {e}")

    # Add FAQ Tool
    try:
        add_faq_tool = AddFAQTool()
        tool_registry[add_faq_tool.name] = add_faq_tool
        print(f"Инструмент '{add_faq_tool.name}' инициализирован.")
    except Exception as e:
        print(f"Ошибка при инициализации AddFAQTool: {e}")

    # Update FAQ Tool
    try:
        update_faq_tool = UpdateFAQTool()
        tool_registry[update_faq_tool.name] = update_faq_tool
        print(f"Инструмент '{update_faq_tool.name}' инициализирован.")
    except Exception as e:
        print(f"Ошибка при инициализации UpdateFAQTool: {e}")

    # Delete FAQ Tool
    try:
        delete_faq_tool = DeleteFAQTool()
        tool_registry[delete_faq_tool.name] = delete_faq_tool
        print(f"Инструмент '{delete_faq_tool.name}' инициализирован.")
    except Exception as e:
        print(f"Ошибка при инициализации DeleteFAQTool: {e}")

    if not tool_registry:
        print("Внимание: Ни один инструмент не был успешно инициализирован!")

    # --- Сборка Графа ---
    app = build_graph(llm)  # Использует обновленный build_graph
    return app
