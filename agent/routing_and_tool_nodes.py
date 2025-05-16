import logging
import json
import uuid
from typing import Dict, Any, List, Literal
import asyncio

from langgraph.graph import END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage

from .state import AgentState
from .tools_registry import tool_registry, tools_list
from .node_utils import (
    llm_instance, 
    trim_messages, 
    extract_tool_call_from_content, 
    TEST_TAVILY_ONLY_MODE,
    LLM_TIMEOUT,
    TOOL_TIMEOUT
)

logger = logging.getLogger(__name__)

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

def analyze_context(state: AgentState) -> Dict[str, Any]:
    """Анализирует текущий контекст диалога."""
    logger.debug("===Analyze Context===")
    # Можно добавить результат анализа в состояние, если он нужен дальше
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

async def router_node(state: AgentState) -> Dict[str, Any]:
    """Анализирует сообщение пользователя и решает, какой инструмент вызвать или генерировать ответ."""
    logger.info("--- Вход в Умный Роутер ---")
    if not llm_instance:
        logger.error(
            "Router Node: LLM не инициализирована!"
        )
        error_ai_msg = AIMessage(
            content="Ошибка конфигурации: LLM не настроена для роутера."
        )
        return {"messages": add_messages(state["messages"], [error_ai_msg])}

    messages = state["messages"]
    last_message = messages[-1]
    logger.info(
        f"Последнее сообщение для роутера: {last_message.content if hasattr(last_message, 'content') else '[No Content]'}"
    )

    if TEST_TAVILY_ONLY_MODE:
        logger.warning("!!! РЕЖИМ ТЕСТИРОВАНИЯ TAVILY API ВКЛЮЧЕН !!!")
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
        router_system_prompt = ROUTER_SYSTEM_PROMPT.format(
            tool_descriptions=tool_descriptions_str
        )
        trimmed_messages = trim_messages(messages, max_history=4)
        messages_for_llm_router = [
            SystemMessage(content=router_system_prompt)
        ] + trimmed_messages

        logger.debug(f"Системный промпт для router LLM: \n{router_system_prompt}")
        logger.debug(f"Сообщения для LLM роутера: {messages_for_llm_router}")
        logger.debug(f"Инструменты для LLM роутера: {formatted_tools_for_router}")

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
        if tool_calls_list_from_attribute:
            final_tool_calls_list = tool_calls_list_from_attribute
            logger.info("Используются tool_calls из атрибута ai_message.tool_calls.")
        elif tool_calls_extracted_from_content:
            final_tool_calls_list = tool_calls_extracted_from_content
            logger.info("Используются tool_calls, извлеченные из ai_message.content.")

        if not final_tool_calls_list and ai_message.content:
            tool_calls_extracted_from_content = extract_tool_call_from_content(
                ai_message.content
            )
            if tool_calls_extracted_from_content:
                final_tool_calls_list = tool_calls_extracted_from_content
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
                }
            else:
                actual_content_for_final_message = ai_message.content
                if isinstance(ai_message.content, str):
                    try:
                        cleaned_content_str = ai_message.content.strip()
                        if cleaned_content_str.startswith("```json"):
                            cleaned_content_str = cleaned_content_str[7:]
                        elif cleaned_content_str.startswith("```"):
                            cleaned_content_str = cleaned_content_str[3:]
                        if cleaned_content_str.endswith("```"):
                            cleaned_content_str = cleaned_content_str[:-3]
                        cleaned_content_str = cleaned_content_str.strip()

                        if cleaned_content_str:
                            parsed_json_content = json.loads(cleaned_content_str)
                            if (
                                isinstance(parsed_json_content, dict)
                                and "content" in parsed_json_content
                            ):
                                actual_content_for_final_message = parsed_json_content[
                                    "content"
                                ]
                                logger.info(
                                    f"Извлечен внутренний 'content' из JSON ответа LLM-роутера: '{str(actual_content_for_final_message)[:100]}...'"
                                )
                            else:
                                logger.info(
                                    "Ответ LLM-роутера был JSON, но без внутреннего ключа 'content'. Используется очищенный JSON как текст."
                                )
                                actual_content_for_final_message = cleaned_content_str
                        else:
                            logger.info(
                                "Content LLM-роутера стал пустым после очистки markdown. Исходный content будет использован."
                            )
                    except json.JSONDecodeError:
                        logger.info(
                            "Content LLM-роутера не валидный JSON (даже после очистки). Исходный content будет использован."
                        )
                    except Exception as e_parse:
                        logger.warning(
                            f"Неожиданная ошибка при парсинге content от LLM-роутера: {e_parse}. Исходный content будет использован."
                        )

                final_ai_message_to_add = AIMessage(
                    content=str(
                        actual_content_for_final_message
                    ),  
                    id=ai_message.id,
                    response_metadata=ai_message.response_metadata,
                )
                logger.info(
                    f"LLM-роутер сгенерировал текстовый ответ (финальный для добавления): '{str(final_ai_message_to_add.content)[:200]}...'"
                )
                return {"messages": add_messages(messages, [final_ai_message_to_add])}

        message_to_add_to_history = ai_message

        if final_tool_calls_list and not tool_calls_list_from_attribute:
            logger.info(
                "Создание нового AIMessage, так как tool_calls были извлечены из ai_message.content."
            )
            new_response_metadata = (
                ai_message.response_metadata.copy()
                if ai_message.response_metadata
                else {}
            )
            new_response_metadata["finish_reason"] = "tool_calls"

            new_ai_message_with_tool_calls = AIMessage(
                content="",
                tool_calls=final_tool_calls_list, 
                id=ai_message.id, 
                response_metadata=new_response_metadata,
            )
            message_to_add_to_history = new_ai_message_with_tool_calls
            logger.info(
                f"Новый AIMessage для истории (с исправленным finish_reason): {message_to_add_to_history}"
            )

        first_tool_call = final_tool_calls_list[0]
        tool_name_chosen = first_tool_call.get("name")
        if tool_name_chosen not in tool_registry:
            logger.error(
                f"LLM выбрала неизвестный инструмент: {tool_name_chosen}. Инструменты в реестре: {list(tool_registry.keys())}"
            )
            error_content = (
                f"Ошибка: попытка вызова неизвестного инструмента '{tool_name_chosen}'."
            )
            error_tool_msg = ToolMessage(
                content=error_content,
                tool_call_id=first_tool_call.get(
                    "id", f"error_unknown_tool_{tool_name_chosen}"
                ),
            )
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

async def tool_executor_node(state: AgentState) -> Dict[str, Any]:
    """Выполняет вызванные инструменты."""
    logger.info("--- Вход в Исполнитель Инструментов ---")
    messages = state["messages"]
    last_message = messages[-1]

    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        logger.warning(
            "Tool Executor: Последнее сообщение не AIMessage или не содержит tool_calls. Пропуск."
        )
        return {} 

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
                    name=tool_name, 
                    additional_kwargs={"is_error": True}, 
                )
            )
            continue

        if hasattr(tool_to_execute, "args_schema") and tool_to_execute.args_schema:
            try:
                validated_input = tool_to_execute.args_schema(**tool_input)
                tool_input_for_invoke = validated_input.dict() 
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
                continue
        else:
            tool_input_for_invoke = tool_input
            if not isinstance(tool_input_for_invoke, dict):
                logger.warning(
                    f"Вход для инструмента '{tool_name}' не словарь и нет args_schema. Передаю как есть: {tool_input_for_invoke}"
                )

        try:
            async with asyncio.timeout(TOOL_TIMEOUT):
                if hasattr(tool_to_execute, "_arun"):
                    logger.debug(
                        f"Асинхронный вызов '_arun' для инструмента '{tool_name}'"
                    )
                    tool_output = await tool_to_execute.ainvoke(tool_input_for_invoke)
                else:
                    logger.debug(
                        f"Синхронный вызов '_run' для инструмента '{tool_name}'"
                    )
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
    return {}

def tool_output_guardrails_node(state: AgentState) -> Dict[str, Any]:
    """Проверяет вывод инструментов."""
    logger.info("--- Вход в Tool Output Guardrails ---")
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