import logging
import json
import operator
import os
import functools
import uuid
from typing import Annotated, Optional, List, Dict, Any, Sequence, Tuple, Union, Literal
import asyncio
import re

from langgraph.graph import END
from langgraph.graph.message import AnyMessage, add_messages

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    FunctionMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatYandexGPT
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

# --- Импорт состояния и реестра инструментов ---
from .state import AgentState
from .tools_registry import (
    tool_registry,
    tools_list,
)  # Используем tools_list для llm.bind_tools

# --- Настройка логгера ---
logger = logging.getLogger(__name__)

# --- Глобальная переменная для LLM (устанавливается в llm_setup.py) ---
llm_instance: Optional[ChatOpenAI] = None

# Системный промпт для генератора ответа (из вашего agent_executor)
SYSTEM_PROMPT_RESPONSE_GENERATOR = """Ты - полезный AI ассистент. Отвечай на последний вопрос пользователя ясно и по делу,
учитывая всю предыдущую историю диалога, включая результаты вызова инструментов (ToolMessage).
Основывай свой ответ на результатах инструментов, если они релевантны.
Если последний запрос был на выполнение действия (add/update/delete) и он выполнен успешно (видно из ToolMessage),
просто подтверди это кратко.
"""

# Сокращенный системный промпт для роутера
ROUTER_SYSTEM_PROMPT = """Ты - роутер запросов пользователя. Твоя цель - выбрать наилучшее действие: вызвать один из доступных инструментов или сгенерировать прямой текстовый ответ.

Доступные инструменты:
{tool_descriptions}

Правила выбора:
1. Проанализируй ПОСЛЕДНИЙ запрос пользователя.
2. Если запрос **однозначно** относится к **Yandex Cloud** (например, 'как настроить ВМ ЯО', 'ошибки Yandex Cloud', 'документация YC', 'сервисы Yandex Cloud'), ты **ОБЯЗАН** вызвать инструмент `tavily_yandexcloud_search`. Используй полный текст запроса пользователя как аргумент `query` для этого инструмента.
3. Если запрос не о Yandex Cloud, но может быть найден во **внутренней базе знаний (FAQ)** (например, вопросы о специфике нашего проекта, внутренних процессах, часто задаваемые вопросы, которые мы сами добавляли), вызови `search_faq`.
4. Если запрос касается **истории диалогов** или ранее обсуждавшихся тем, вызови `search_chat_history`.
5. Для запросов на **добавление, обновление или удаление** записей в FAQ используй соответственно `add_faq`, `update_faq`, `delete_faq`. Требуется явное намерение пользователя и наличие необходимых данных (вопрос/ответ, ID).
6. Во всех остальных случаях, когда ни один инструмент не подходит, или после получения результата от инструмента, **сгенерируй прямой, краткий и понятный ответ** для пользователя.

Твой ответ ДОЛЖЕН быть либо JSON объект в поле `tool_calls` сообщения AIMessage (для вызова инструмента), либо просто текст в поле `content` сообщения AIMessage (для прямого ответа).
НЕ ДОБАВЛЯЙ НИКАКОГО ЛИШНЕГО ТЕКСТА ИЛИ ПОЯСНЕНИЙ, КРОМЕ JSON ДЛЯ tool_calls или финального текстового ответа.
"""

# Флаг для тестирования только Tavily API без вызовов LLM (из вашего agent_executor)
# Установим в False, чтобы граф работал в обычном режиме
TEST_TAVILY_ONLY_MODE = False

# Таймаут для вызовов LLM (в секундах)
LLM_TIMEOUT = 60
# Таймаут для выполнения инструментов (в секундах)
TOOL_TIMEOUT = 90  # Например, 90 секунд на выполнение инструмента


def set_node_llm(llm_to_set: ChatOpenAI):
    """Устанавливает глобальную LLM для использования в узлах."""
    global llm_instance
    llm_instance = llm_to_set
    logger.debug(f"LLM ('{type(llm_instance)}') установлена для graph_nodes.")


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
        # В LangGraph для завершения графа из узла используется специальное значение в `next_node` или raising an exception.
        # Однако, условные ребра должны быть настроены на `END`.
        # Правильнее будет вернуть состояние, которое приведет к END через условное ребро.
        # Здесь мы просто добавляем сообщение об ошибке. Условное ребро `should_respond` должно обработать это.
        return {"messages": add_messages(messages, [error_ai_msg])}

    user_input = messages[-1].content
    if not user_input.strip():
        logger.warning("Input Guardrails: Пустое сообщение от пользователя.")
        error_ai_msg = AIMessage(content="Пожалуйста, введите ваш вопрос.")
        return {"messages": add_messages(messages, [error_ai_msg])}

    logger.debug("Input Guardrails: Проверка пройдена.")
    return {}  # Если все ок, не меняем состояние.


def analyze_context(state: AgentState) -> Dict[str, Any]:
    """Анализирует текущий контекст диалога."""
    logger.debug("===Analyze Context===")
    # Здесь может быть логика анализа истории, извлечения сущностей и т.д.
    # Можно использовать инструмент context_analyzer
    context_analysis_result = "Контекст проанализирован."
    logger.debug(f"Context Analysis Result: {context_analysis_result}")
    # Можно добавить результат анализа в состояние, если он нужен дальше
    # return {"context_analysis": context_analysis_result}
    return {}


def should_respond(state: AgentState) -> Literal["continue", "__end__"]:
    """Проверяет, нужно ли продолжать или можно закончить."""
    logger.debug("===Checking Should Respond===")
    messages = state["messages"]
    if not messages:
        logger.warning("Should Respond: Список сообщений пуст. ENDING.")
        return END

    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
        logger.debug(
            "Should Respond: Last message is AIMessage without tool calls. ENDING."
        )
        return END
    if any(
        isinstance(m, AIMessage) and "Ошибка:" in m.content
        for m in messages
        if not (
            isinstance(m, ToolMessage) or (isinstance(m, AIMessage) and m.tool_calls)
        )
    ):
        logger.warning("Should Respond: Found error message. ENDING.")
        return END
    if isinstance(last_message, ToolMessage) and last_message.additional_kwargs.get(
        "is_error", False
    ):
        logger.debug("Should Respond: Last message is ToolMessage with error. ENDING.")
        return END

    logger.debug("Should Respond: Continuing.")
    return "continue"


# Функция для обрезки истории сообщений
def trim_messages(
    messages: Sequence[BaseMessage], max_history: int = 5
) -> List[BaseMessage]:
    """
    Обрезает историю сообщений, оставляя только последние `max_history` сообщений,
    исключая SystemMessage.
    """
    # Убираем системные сообщения, так как они добавляются отдельно перед каждым вызовом LLM
    filtered = [m for m in messages if not isinstance(m, SystemMessage)]
    # Возвращаем последние N сообщений
    return filtered[-max_history:]


# --- Роутер: Выбор между вызовом инструмента и генерацией ответа --- #
async def router_node(state: AgentState) -> Dict[str, Any]:
    """(Из вашего agent_executor, адаптированный)
    Анализирует сообщение пользователя и решает, какой инструмент вызвать или генерировать ответ.
    """
    logger.info("--- Вход в Умный Роутер ---")
    global llm_instance  # Используем глобальную LLM
    if not llm_instance:
        logger.error(
            "Router Node: LLM не инициализирована! Убедитесь, что set_node_llm была вызвана."
        )
        error_ai_msg = AIMessage(
            content="Ошибка конфигурации: LLM не настроена для роутера."
        )
        # LangGraph ожидает, что условное ребро будет возвращать ключ из словаря ребер.
        # Поэтому просто добавление "next_node" здесь не сработает без соответствующей настройки ребра.
        # Правильнее вернуть AIMessage, и пусть условное ребро решит.
        return {"messages": add_messages(state["messages"], [error_ai_msg])}

    messages = state["messages"]
    last_message = messages[-1]
    logger.info(
        f"Последнее сообщение для роутера: {last_message.content if hasattr(last_message, 'content') else '[No Content]'}"
    )

    if TEST_TAVILY_ONLY_MODE:
        logger.warning("!!! РЕЖИМ ТЕСТИРОВАНИЯ TAVILY API ВКЛЮЧЕН !!!")
        # Ищем инструмент tavily_yandexcloud_search в реестре
        tavily_tool = tool_registry.get("tavily_yandexcloud_search")
        if not tavily_tool:
            logger.error(
                f"Инструмент 'tavily_yandexcloud_search' не найден в реестре. Тест Tavily невозможен."
            )
            error_ai_msg = AIMessage(
                content="Ошибка: тестовый инструмент tavily_yandexcloud_search не зарегистрирован."
            )
            return {"messages": add_messages(messages, [error_ai_msg])}

        tool_name_to_test = tavily_tool.name
        fake_tool_call_id = f"call_test_tavily_{uuid.uuid4()}"
        query_content = (
            last_message.content
            if isinstance(last_message.content, str)
            else "test query"
        )

        fake_ai_message = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": fake_tool_call_id,
                    "name": tool_name_to_test,
                    "args": {"query": query_content},
                }
            ],
        )
        logger.info(f"Фейковый AIMessage для теста Tavily: {fake_ai_message}")
        updated_messages = add_messages(messages, [fake_ai_message])
        return {"messages": updated_messages}

    if not tool_registry:
        logger.error(
            "Реестр инструментов (tool_registry) пуст! Невозможно привязать инструменты к LLM."
        )
        error_ai_msg = AIMessage(
            content="Ошибка конфигурации: нет доступных инструментов."
        )
        return {"messages": add_messages(messages, [error_ai_msg])}

    if isinstance(last_message, ToolMessage):
        logger.info(
            "Последнее сообщение - результат инструмента. Переход к генерации ответа."
        )
        # Не добавляем AIMessage, граф перейдет к generate_response на основе этого.
        # Просто возвращаем текущее состояние, чтобы условное ребро сработало.
        return {}

    logger.info("Запрос к LLM-роутеру с ЯВНО ПЕРЕДАННЫМИ инструментами...")
    try:
        formatted_tools_for_router = []
        tool_descriptions_for_prompt_list = []
        for tool_instance in tools_list:
            if hasattr(tool_instance, "args_schema") and tool_instance.args_schema:
                parameters = tool_instance.args_schema.schema()
            else:
                parameters = {}
            formatted_tools_for_router.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_instance.name,
                        "description": tool_instance.description[:100],
                        "parameters": parameters,
                    },
                }
            )
            tool_descriptions_for_prompt_list.append(
                f"- {tool_instance.name}: {tool_instance.description[:100]}"
            )

        if not formatted_tools_for_router:
            logger.error(
                "Не удалось сформировать описания инструментов для LLM-роутера (список пуст)."
            )
            error_ai_msg = AIMessage(
                content="Внутренняя ошибка: не удалось подготовить инструменты для LLM."
            )
            return {"messages": add_messages(messages, [error_ai_msg])}

        tool_descriptions_str = "\n".join(tool_descriptions_for_prompt_list)
        # Используем сокращенный системный промпт
        router_system_prompt = ROUTER_SYSTEM_PROMPT.format(
            tool_descriptions=tool_descriptions_str
        )

        # Обрезаем историю сообщений перед отправкой в LLM
        trimmed_messages = trim_messages(messages, max_history=4)

        messages_for_llm_router = [
            SystemMessage(content=router_system_prompt)
        ] + trimmed_messages

        logger.debug(f"Системный промпт для router LLM: \n{router_system_prompt}")
        logger.debug(f"Сообщения для LLM роутера: {messages_for_llm_router}")
        logger.debug(f"Инструменты для LLM роутера: {formatted_tools_for_router}")

        # Логирование размера payload перед вызовом
        try:
            payload_size = len(
                json.dumps(
                    [
                        m.dict() if hasattr(m, "dict") else str(m)
                        for m in messages_for_llm_router
                    ]
                    + formatted_tools_for_router
                )
            )
            logger.info(
                f"Payload size before LLM call (router): {payload_size} bytes, messages count: {len(messages_for_llm_router)}, tools count: {len(formatted_tools_for_router)}"
            )
        except Exception as log_e:
            logger.warning(
                f"Ошибка при логировании размера payload в router_node: {log_e}"
            )

        # Добавляем таймаут на вызов LLM
        try:
            async with asyncio.timeout(LLM_TIMEOUT):
                ai_message: AIMessage = await llm_instance.ainvoke(
                    messages_for_llm_router, tools=formatted_tools_for_router
                )
            logger.info(f"Ответ LLM-роутера (сырой объект): {ai_message}")

        except asyncio.TimeoutError:
            logger.error(f"Таймаут ({LLM_TIMEOUT} сек) при вызове LLM-роутера.")
            error_ai_msg = AIMessage(content="Извините, роутер LLM не ответил вовремя.")
            return {"messages": add_messages(messages, [error_ai_msg])}
        except Exception as e:
            logger.error(
                f"Ошибка в router_node при вызове LLM с инструментами: {e}",
                exc_info=True,
            )
            error_ai_msg = AIMessage(
                content=f"Произошла ошибка при попытке выбрать действие: {e}"
            )
            return {"messages": add_messages(messages, [error_ai_msg])}

        tool_calls_list_from_attribute = None
        if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
            tool_calls_list_from_attribute = ai_message.tool_calls
            logger.info("Извлечены tool_calls из ai_message.tool_calls атрибута")

        tool_calls_extracted_from_content = None
        if (
            not tool_calls_list_from_attribute
            and ai_message.content
            and isinstance(ai_message.content, str)
        ):
            logger.info(
                "ai_message.tool_calls пусто, но content присутствует. Попытка парсинга content для tool_calls."
            )
            try:
                content_str = ai_message.content.strip()
                if content_str.startswith("```json"):
                    content_str = content_str[7:]
                elif content_str.startswith("```"):
                    content_str = content_str[3:]
                if content_str.endswith("```"):
                    content_str = content_str[:-3]
                content_str = content_str.strip()
                if not content_str:
                    raise ValueError("Content стал пустым после очистки от markdown.")
                parsed_content = json.loads(content_str)
                if isinstance(parsed_content, list) and parsed_content:
                    potential_tool_call_data = parsed_content[0]
                    if (
                        isinstance(potential_tool_call_data, dict)
                        and potential_tool_call_data.get("type") == "function"
                        and isinstance(potential_tool_call_data.get("function"), dict)
                    ):
                        func_details = potential_tool_call_data["function"]
                        tool_name = func_details.get("name")
                        raw_arguments = func_details.get("arguments")
                        tool_id = potential_tool_call_data.get(
                            "id", f"parsed_call_{uuid.uuid4()}"
                        )
                        actual_tool_args = {}
                        if isinstance(raw_arguments, str):
                            try:
                                actual_tool_args = json.loads(raw_arguments)
                            except json.JSONDecodeError:
                                logger.error(
                                    f"Ошибка декодирования JSON из строки arguments: {raw_arguments}"
                                )
                        elif isinstance(raw_arguments, dict):
                            actual_tool_args = raw_arguments
                        else:
                            logger.warning(
                                f"Аргументы инструмента '{tool_name}' имеют неожиданный тип: {type(raw_arguments)}."
                            )
                        if tool_name:
                            tool_calls_extracted_from_content = [
                                {
                                    "id": tool_id,
                                    "name": tool_name,
                                    "args": actual_tool_args,
                                }
                            ]
                            logger.info(
                                f"Успешно распарсен tool_call из content: {tool_calls_extracted_from_content}"
                            )
                        else:
                            logger.warning(
                                "Распарсенный tool_call из content не содержит имени инструмента."
                            )
                    else:
                        logger.info(
                            "Распарсенный content не соответствует структуре tool_call."
                        )
                else:
                    logger.info("Распарсенный content не является списком или пуст.")
            except Exception as e:
                logger.error(
                    f"Ошибка при парсинге tool_call из content: {e}", exc_info=True
                )

        final_tool_calls_list = None
        source_of_tool_calls = None
        if tool_calls_list_from_attribute:
            final_tool_calls_list = tool_calls_list_from_attribute
            source_of_tool_calls = "attribute"
            logger.info("Используются tool_calls из атрибута ai_message.tool_calls.")
        elif tool_calls_extracted_from_content:
            final_tool_calls_list = tool_calls_extracted_from_content
            source_of_tool_calls = "content"
            logger.info("Используются tool_calls, извлеченные из ai_message.content.")

        if not final_tool_calls_list and ai_message.content:
            tool_calls_extracted_from_content = extract_tool_call_from_content(
                ai_message.content
            )
            if tool_calls_extracted_from_content:
                final_tool_calls_list = tool_calls_extracted_from_content
                source_of_tool_calls = "manual_content_parser"
                logger.info("Tool call извлечён эвристическим парсером из content.")

        if not final_tool_calls_list:
            logger.warning(
                f"Не удалось извлечь tool_call. Сырой content: {ai_message.content}"
            )
            if (
                hasattr(ai_message, "response_metadata")
                and ai_message.response_metadata.get("finish_reason") == "tool_calls"
                and not (hasattr(ai_message, "tool_calls") and ai_message.tool_calls)
            ):
                logger.warning(
                    f"LLM-роутер ({ai_message.id if hasattr(ai_message, 'id') else 'N/A'}) вернул finish_reason='tool_calls', но поле tool_calls пустое. Не добавляем AIMessage."
                )
                return {
                    "messages": messages
                }  # Возвращаем старые сообщения, без этого некорректного AIMessage
            else:
                logger.info(
                    f"LLM-роутер сгенерировал текстовый ответ: '{ai_message.content}'"
                )
                return {"messages": add_messages(messages, [ai_message])}

        message_to_add_to_history = ai_message

        # Условие: если final_tool_calls_list не пуст Изначально tool_calls в атрибуте были пусты
        # это означает, что final_tool_calls_list был получен из ai_message.content
        if final_tool_calls_list and not tool_calls_list_from_attribute:
            logger.info(
                "Создание нового AIMessage, так как tool_calls были извлечены из ai_message.content."
            )

            # Создаем копию response_metadata или пустой словарь, если его нет
            new_response_metadata = (
                ai_message.response_metadata.copy()
                if ai_message.response_metadata
                else {}
            )
            # Устанавливаем корректный finish_reason
            new_response_metadata["finish_reason"] = "tool_calls"

            new_ai_message_with_tool_calls = AIMessage(
                content="",  # Контент должен быть пуст, т.к. все ушло в tool_calls
                tool_calls=final_tool_calls_list,  # Заполняем извлеченными данными
                id=ai_message.id,  # Копируем ID из оригинального сообщения
                response_metadata=new_response_metadata,  # Используем обновленный metadata
            )
            message_to_add_to_history = new_ai_message_with_tool_calls
            logger.info(
                f"Новый AIMessage для истории (с исправленным finish_reason): {message_to_add_to_history}"
            )

        # Проверка, что выбранный инструмент существует в реестре
        # Это важно перед добавлением message_to_add_to_history, если он содержит tool_call к несуществующему инструменту
        first_tool_call = final_tool_calls_list[0]
        tool_name_chosen = first_tool_call.get("name")
        if tool_name_chosen not in tool_registry:
            logger.error(
                f"LLM выбрала неизвестный инструмент: {tool_name_chosen}. Инструменты в реестре: {list(tool_registry.keys())}"
            )
            error_content = (
                f"Ошибка: попытка вызова неизвестного инструмента '{tool_name_chosen}'."
            )
            # Создаем ToolMessage с ошибкой, который будет соответствовать этому AIMessage
            error_tool_msg = ToolMessage(
                content=error_content,
                tool_call_id=first_tool_call.get(
                    "id", f"error_unknown_tool_{tool_name_chosen}"
                ),
            )
            # Добавляем и AIMessage (возможно, некорректный) и ToolMessage с ошибкой
            return {
                "messages": add_messages(
                    messages, [message_to_add_to_history, error_tool_msg]
                )
            }

        updated_messages = add_messages(messages, [message_to_add_to_history])
        return {"messages": updated_messages}

    except Exception as e:
        logger.error(
            f"Ошибка в router_node при вызове LLM с инструментами: {e}", exc_info=True
        )
        error_ai_msg = AIMessage(
            content=f"Произошла ошибка при попытке выбрать действие: {e}"
        )
        return {"messages": add_messages(messages, [error_ai_msg])}


# --- Узел выполнения инструментов --- (из вашего agent_executor, адаптированный)
async def tool_executor_node(state: AgentState) -> Dict[str, Any]:
    """Выполняет вызванные инструменты."""
    logger.info("--- Вход в Исполнитель Инструментов ---")
    messages = state["messages"]
    last_message = messages[-1]

    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        logger.warning(
            "Tool Executor: Последнее сообщение не AIMessage или не содержит tool_calls. Пропуск."
        )
        return {}  # Ничего не делаем, если нет вызовов инструментов

    tool_messages: List[ToolMessage] = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_input = tool_call["args"]
        tool_call_id = tool_call["id"]

        logger.info(
            f"Вызов инструмента '{tool_name}' с ID '{tool_call_id}' и вводом: {tool_input}"
        )
        tool_to_execute = tool_registry.get(tool_name)

        if not tool_to_execute:
            logger.error(f"Инструмент '{tool_name}' не найден в реестре.")
            tool_messages.append(
                ToolMessage(
                    content=f"Ошибка: Инструмент '{tool_name}' не найден.",
                    tool_call_id=tool_call_id,
                    name=tool_name,  # Добавляем имя для консистентности
                    additional_kwargs={"is_error": True},  # Помечаем как ошибку
                )
            )
            continue

        # Проверка аргументов инструмента (из вашего agent_executor)
        if hasattr(tool_to_execute, "args_schema") and tool_to_execute.args_schema:
            try:
                validated_input = tool_to_execute.args_schema(**tool_input)
                tool_input_for_invoke = validated_input.dict()  # Pydantic model to dict
                logger.debug(
                    f"Инструмент '{tool_name}', валидированный ввод: {tool_input_for_invoke}"
                )
            except Exception as validation_error:
                logger.error(
                    f"Ошибка валидации ввода для инструмента '{tool_name}': {validation_error}",
                    exc_info=True,
                )
                tool_messages.append(
                    ToolMessage(
                        content=f"Ошибка валидации аргументов для '{tool_name}': {validation_error}",
                        tool_call_id=tool_call_id,
                        name=tool_name,
                        additional_kwargs={"is_error": True},
                    )
                )
                continue  # Переходим к следующему вызову инструмента, если валидация не прошла
        else:
            # Если схемы нет, передаем как есть (но это должно быть обработано в инструменте)
            tool_input_for_invoke = tool_input
            if not isinstance(tool_input_for_invoke, dict):
                logger.warning(
                    f"Вход для инструмента '{tool_name}' не словарь и нет args_schema. Передаю как есть: {tool_input_for_invoke}"
                )

        try:
            # Добавляем таймаут на выполнение инструмента
            async with asyncio.timeout(TOOL_TIMEOUT):
                # Асинхронный вызов инструмента, если он поддерживает _arun
                if hasattr(tool_to_execute, "_arun"):
                    logger.debug(
                        f"Асинхронный вызов '_arun' для инструмента '{tool_name}'"
                    )
                    tool_output = await tool_to_execute.ainvoke(tool_input_for_invoke)
                else:
                    logger.debug(
                        f"Синхронный вызов '_run' для инструмента '{tool_name}'"
                    )
                    # Выполняем синхронный код в потоке, чтобы не блокировать asyncio
                    tool_output = await asyncio.to_thread(
                        tool_to_execute.invoke, tool_input_for_invoke
                    )

            tool_messages.append(
                ToolMessage(
                    content=str(tool_output), tool_call_id=tool_call_id, name=tool_name
                )
            )
            logger.info(f"Инструмент '{tool_name}' вернул: {str(tool_output)[:200]}...")

        except asyncio.TimeoutError:
            logger.error(
                f"Таймаут ({TOOL_TIMEOUT} сек) при выполнении инструмента '{tool_name}'."
            )
            tool_messages.append(
                ToolMessage(
                    content=f"Ошибка: Инструмент '{tool_name}' не завершился вовремя ({TOOL_TIMEOUT} сек).",
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    additional_kwargs={"is_error": True, "error_type": "TimeoutError"},
                )
            )
        except Exception as e:
            logger.error(
                f"Ошибка при выполнении инструмента '{tool_name}': {e}", exc_info=True
            )
            tool_messages.append(
                ToolMessage(
                    content=f"Ошибка при выполнении инструмента '{tool_name}': {e}",
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    additional_kwargs={"is_error": True},
                )
            )

    if tool_messages:
        return {"messages": add_messages(messages, tool_messages)}
    return (
        {}
    )  # Возвращаем пустой словарь, если не было успешных/ошибочных tool_messages для добавления


def tool_output_guardrails_node(state: AgentState) -> Dict[str, Any]:
    """(Из вашего agent_executor) Проверяет вывод инструментов."""
    logger.info("--- Вход в Tool Output Guardrails ---")
    # TODO: Реализовать более сложную логику проверки вывода инструментов
    # Пока что просто логируем, если есть ошибки, помеченные в tool_executor
    messages = state["messages"]
    last_message = messages[-1]
    if isinstance(last_message, ToolMessage) and last_message.additional_kwargs.get(
        "is_error"
    ):
        logger.warning(
            f"Tool Output Guardrails: Обнаружена ошибка в выводе инструмента (ID: {last_message.tool_call_id}): {last_message.content}"
        )
    else:
        logger.debug(
            "Tool Output Guardrails: Проверка вывода инструментов пройдена (или ошибок не было)."
        )
    return {}


def handle_tool_result_node(state: AgentState) -> Dict[str, Any]:
    """(Из вашего agent_executor) Решает, что делать после инструмента.
    В новой структуре графа этот узел может быть менее критичен, так как
    условное ребро после tool_output_guardrails может вести сразу к response_generator или END.
    Пока оставим как есть, он может добавить AIMessage с результатами или ошибкой для LLM.
    """
    logger.info("--- Вход в Обработчик Результатов Инструмента ---")
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, ToolMessage):
        # Если инструмент вернул ошибку, которую мы пометили
        if last_message.additional_kwargs.get("is_error"):
            logger.warning(
                f"Обработка результата: Инструмент {last_message.name} (ID: {last_message.tool_call_id}) вернул ошибку. Переход к генерации ответа (который должен сообщить об ошибке)."
            )
            # Не меняем next_node, граф пойдет по обычному пути к response_generator
            return {}
        else:
            logger.info(
                f"Обработка результата: Инструмент {last_message.name} (ID: {last_message.tool_call_id}) отработал. Переход к генерации ответа."
            )
            # Не меняем next_node
            return {}
    logger.debug(
        "Обработка результата: последнее сообщение не ToolMessage. Ничего не делаем."
    )
    return {}


async def response_generator_node(state: AgentState) -> Dict[str, Any]:
    """(Из вашего agent_executor, адаптированный) Генерирует финальный ответ пользователю."""
    logger.info("--- Вход в Генератор Ответа ---")
    global llm_instance
    if not llm_instance:
        logger.error("Response Generator: LLM не инициализирована!")
        # Возвращаем AIMessage с ошибкой, граф должен завершиться после этого (через output_guardrails -> END)
        error_ai_msg = AIMessage(
            content="Ошибка конфигурации: LLM не настроена для генерации ответа."
        )
        return {"messages": add_messages(state["messages"], [error_ai_msg])}

    # Обрезаем историю сообщений перед отправкой в LLM
    trimmed_messages = trim_messages(state["messages"], max_history=5)

    messages_for_llm = [
        SystemMessage(content=SYSTEM_PROMPT_RESPONSE_GENERATOR),
        # Убираем системные сообщения из предыдущей истории, если они там есть
        # Используем обрезанные сообщения
        *trimmed_messages,
    ]

    logger.debug(f"Сообщения для генератора ответа LLM: {messages_for_llm}")
    try:
        # Не используем bind_tools здесь, просто генерируем ответ
        # Логирование размера payload перед вызовом
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

        # Добавляем таймаут на вызов LLM
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
    """(Из вашего agent_executor) Проверяет финальный ответ перед отправкой пользователю."""
    logger.info("--- Вход в Output Guardrails ---")
    messages = state["messages"]
    if not messages or not isinstance(messages[-1], AIMessage):
        logger.warning(
            "Output Guardrails: Последнее сообщение не AIMessage или пусто. Нечего проверять."
        )
        # Если последнее сообщение - это не AIMessage, то, возможно, предыдущий узел (response_generator)
        # вернул ошибку, которая уже добавлена как AIMessage.
        # Или граф идет по ошибочной ветке. Просто возвращаем состояние как есть.
        return {}

    last_ai_message = messages[-1]
    if not last_ai_message.content or not last_ai_message.content.strip():
        logger.warning(
            "Output Guardrails: Финальный ответ AIMessage пустой. Замена на fallback."
        )
        fallback_response = AIMessage(
            content="Извините, я не смог сформировать внятный ответ на ваш запрос."
        )
        # Заменяем последнее (пустое) AIMessage на fallback
        updated_messages = list(state["messages"][:-1]) + [fallback_response]
        return {"messages": updated_messages}

    logger.debug(
        f"Output Guardrails: Финальный ответ проверен: {last_ai_message.content[:100]}..."
    )
    return {}  # Возвращаем пустой словарь, если изменений нет


# --- Старые узлы из graph_nodes.py, которые не были в agent_executor.py ---
# analyze_context и should_respond.
# Их нужно либо удалить, либо адаптировать, если их логика все еще нужна.
# Пока что закомментируем их, чтобы не было конфликтов имен, если они не используются в новой структуре графа.

# def analyze_context(state: AgentState) -> Dict[str, Any]: ... (старая версия)
# def should_respond(state: AgentState) -> Literal["continue", END]: ... (старая версия)


def extract_tool_call_from_content(content: str):
    # 1. Попробуй как есть (валидный JSON)
    try:
        parsed = json.loads(content)
        # Если это список с нужной структурой — возвращаем
        if isinstance(parsed, list) and parsed and "name" in parsed[0]:
            return parsed
        # Если это dict с ключами tool_call или function_call
        if isinstance(parsed, dict) and (
            "tool_call" in parsed or "function_call" in parsed
        ):
            tc = parsed.get("tool_call") or parsed.get("function_call")
            if isinstance(tc, dict) and "name" in tc:
                return [
                    {
                        "id": tc.get("id", "manual_extracted"),
                        "name": tc["name"],
                        "args": tc.get("args", {}),
                    }
                ]
    except Exception:
        pass

    # 2. Попробуй вытащить JSON из markdown
    match = re.search(r"```(?:json)?(.*)```", content, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1).strip())
            if isinstance(parsed, list) and parsed and "name" in parsed[0]:
                return parsed
        except Exception:
            pass

    # 3. Попробуй найти паттерн "название_инструмента" и "query"
    match = re.search(
        r'(tavily_yandexcloud_search).*?query[":= ]+["\']?([^"\'}\n]+)',
        content,
        re.IGNORECASE,
    )
    if match:
        tool_name = match.group(1)
        query = match.group(2)
        return [
            {"id": f"manual_{tool_name}", "name": tool_name, "args": {"query": query}}
        ]

    # 4. Можно добавить ещё эвристик
    return None
