import logging
import operator
import os
import functools
import uuid  # Для генерации фейкового tool_call_id
from typing import Sequence, Literal, Dict, Any
import json

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

    if isinstance(last_message, ToolMessage):
        logger.info(
            "Последнее сообщение - результат инструмента. Переход к генерации ответа."
        )
        return {"next_node": "generate_response"}

    logger.info("Запрос к LLM-роутеру с ЯВНО ПЕРЕДАННЫМИ инструментами...")
    try:
        # Формируем описания инструментов для явной передачи и для системного промпта
        formatted_tools_for_router = []
        tool_descriptions_for_prompt_list = []
        for tool_name, tool_instance in tool_registry.items():
            formatted_tools_for_router.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_instance.name,
                        "description": tool_instance.description,
                        "parameters": (
                            tool_instance.args_schema.schema()
                            if tool_instance.args_schema
                            else {}
                        ),
                    },
                }
            )
            tool_descriptions_for_prompt_list.append(
                f"- {tool_instance.name}: {tool_instance.description}"
            )

        if not formatted_tools_for_router:
            logger.error(
                "Не удалось сформировать описания инструментов для LLM-роутера (список пуст)."
            )
            error_ai_msg = AIMessage(
                content="Внутренняя ошибка: не удалось подготовить инструменты для LLM."
            )
            return {
                "next_node": "generate_response",
                "messages": messages + [error_ai_msg],
            }

        tool_descriptions_str = "\n".join(tool_descriptions_for_prompt_list)

        # Новый ОЧЕНЬ ДИРЕКТИВНЫЙ системный промпт для роутера
        router_system_prompt = f"""Твоя единственная задача - маршрутизация запроса пользователя.

Доступные инструменты:
{tool_descriptions_str}

Правила:
1. Проанализируй ПОСЛЕДНИЙ запрос пользователя.
2. Если запрос касается Yandex Cloud (например, 'как настроить ВМ', 'сервисы Yandex Cloud', 'ошибки Yandex Cloud', 'подключение к Yandex Cloud', 'документация Yandex Cloud'):
   Ты ОБЯЗАН вызвать инструмент 'tavily_yandexcloud_search'.
   Аргументом 'query' для этого инструмента должен быть ПОЛНЫЙ ТЕКСТ последнего запроса пользователя.
   Твой ответ ДОЛЖЕН быть ТОЛЬКО JSON объект в поле 'tool_calls' сообщения. Этот JSON должен быть массивом, содержащим один объект вызова функции.
   Пример формата ТОЛЬКО для 'tool_calls', если пользователь спросил "Как настроить VPN в Yandex Cloud?":
   `[ {{"id": "call_random_id_123", "type": "function", "function": {{"name": "tavily_yandexcloud_search", "arguments": "{{\"query\": \"Как настроить VPN в Yandex Cloud?\"}}"}}}} ]`
   НЕ ДОБАВЛЯЙ НИКАКОГО ДРУГОГО ТЕКСТА, ПОЯСНЕНИЙ ИЛИ КОММЕНТАРИЕВ В ТВОЙ ОТВЕТ. ТОЛЬКО `tool_calls`.
3. Если запрос НЕ о Yandex Cloud:
   Рассмотри другие инструменты ('web_search' для общих вопросов, 'search_faq', 'add_faq', 'update_faq', 'delete_faq' для работы с базой знаний).
   Если подходящий инструмент найден, используй тот же формат ответа ТОЛЬКО с 'tool_calls'.
   Если ни один инструмент не подходит, и ты можешь ответить сам, сгенерируй прямой текстовый ответ. В этом случае в твоем ответе НЕ ДОЛЖНО БЫТЬ поля 'tool_calls'.
"""

        # Подготавливаем сообщения для LLM, добавляя системный промпт
        messages_for_llm_router = [
            SystemMessage(content=router_system_prompt)
        ] + messages

        logger.info(
            f"Передаваемые инструменты в router LLM: {formatted_tools_for_router}"
        )
        logger.info(f"Системный промпт для router LLM: \n{router_system_prompt}")

        # Используем базовый llm объект и передаем tools как kwarg
        ai_message: AIMessage = llm.invoke(
            messages_for_llm_router, tools=formatted_tools_for_router
        )

        logger.info(f"Ответ LLM-роутера (сырой объект): {ai_message}")
        logger.info(f"Тип ai_message: {type(ai_message)}")
        logger.info(f"Контент ai_message: {ai_message.content}")
        logger.info(f"additional_kwargs ai_message: {ai_message.additional_kwargs}")
        logger.info(f"response_metadata ai_message: {ai_message.response_metadata}")
        logger.info(
            f"Есть ли атрибут 'tool_calls' у ai_message: {hasattr(ai_message, "tool_calls")}"
        )
        if hasattr(ai_message, "tool_calls"):
            logger.info(f"Значение ai_message.tool_calls: {ai_message.tool_calls}")
            logger.info(f"Тип ai_message.tool_calls: {type(ai_message.tool_calls)}")
            if ai_message.tool_calls:
                logger.info(f"Количество tool_calls: {len(ai_message.tool_calls)}")

        # --- ИСПРАВЛЕНИЕ ЛОГИКИ ИЗВЛЕЧЕНИЯ TOOL_CALLS + ПАРСИНГ ИЗ CONTENT ---
        tool_calls_list_from_attribute = None
        if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
            tool_calls_list_from_attribute = ai_message.tool_calls
            logger.info("Извлечены tool_calls из ai_message.tool_calls атрибута")

        tool_calls_extracted_from_content = None
        # Пытаемся извлечь из content, ЕСЛИ в атрибуте tool_calls пусто ИЛИ additional_kwargs тоже пуст (на всякий случай)
        # и если content вообще есть
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
                # Удаляем потенциальные markdown code blocks
                if content_str.startswith("```json"):
                    content_str = content_str[7:]  # Удаляем ```json
                elif content_str.startswith("```"):  # Удаляем только ```
                    content_str = content_str[3:]

                if content_str.endswith("```"):
                    content_str = content_str[:-3]
                content_str = content_str.strip()

                if not content_str:
                    raise ValueError("Content стал пустым после очистки от markdown.")

                parsed_content = json.loads(content_str)

                # Ожидаем, что parsed_content - это список словарей (даже если там один вызов)
                if isinstance(parsed_content, list) and parsed_content:
                    # Берем первый элемент, предполагая один вызов инструмента за раз
                    potential_tool_call_data = parsed_content[0]

                    if (
                        isinstance(potential_tool_call_data, dict)
                        and potential_tool_call_data.get("type") == "function"
                        and isinstance(potential_tool_call_data.get("function"), dict)
                    ):

                        func_details = potential_tool_call_data["function"]
                        tool_name = func_details.get("name")
                        # Аргументы могут быть уже словарем или строкой JSON
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
                                # Можно попытаться использовать строку как есть, если инструмент это ожидает,
                                # или вернуть ошибку/пропустить. Для Tavily нужен dict.
                        elif isinstance(raw_arguments, dict):
                            actual_tool_args = raw_arguments
                        else:
                            logger.warning(
                                f"Аргументы инструмента '{tool_name}' имеют неожиданный тип: {type(raw_arguments)}. Ожидался dict или str."
                            )

                        if tool_name:  # Имя инструмента обязательно
                            tool_calls_extracted_from_content = [
                                {
                                    "id": tool_id,  # Langchain ожидает 'id' на верхнем уровне
                                    "name": tool_name,  # 'name' тоже
                                    "args": actual_tool_args,  # 'args' тоже
                                    # 'type' и 'function' были для парсинга, в tool_calls LangChain они не нужны в таком виде
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
                            "Распарсенный content не соответствует ожидаемой структуре tool_call (например, отсутствует type: 'function' или сама 'function')."
                        )
                else:
                    logger.info("Распарсенный content не является списком или пуст.")
            except json.JSONDecodeError:
                logger.info(
                    f"Content не является валидным JSON или не в формате списка: '{ai_message.content[:200]}...'"
                )
            except ValueError as ve:
                logger.info(f"Ошибка при подготовке content для парсинга: {ve}")
            except Exception as e:
                logger.error(
                    f"Неожиданная ошибка при парсинге tool_call из content: {e}",
                    exc_info=True,
                )

        # --- Выбор, какие tool_calls использовать ---
        final_tool_calls_list = None
        source_of_tool_calls = None  # "attribute", "content", "kwargs"

        if tool_calls_list_from_attribute:
            final_tool_calls_list = tool_calls_list_from_attribute
            source_of_tool_calls = "attribute"
            logger.info("Используются tool_calls из атрибута ai_message.tool_calls.")
        elif tool_calls_extracted_from_content:
            final_tool_calls_list = tool_calls_extracted_from_content
            source_of_tool_calls = "content"
            logger.info("Используются tool_calls, извлеченные из ai_message.content.")
        else:
            # Проверяем additional_kwargs как последний вариант (старый код)
            if (
                hasattr(ai_message, "additional_kwargs")
                and "tool_calls" in ai_message.additional_kwargs
            ):
                # ... (код парсинга из additional_kwargs, если он нужен) ...
                # Заглушка, если понадобится:
                # parsed_kwargs_tool_calls = _parse_tool_calls_from_kwargs(ai_message.additional_kwargs["tool_calls"])
                # if parsed_kwargs_tool_calls:
                #    final_tool_calls_list = parsed_kwargs_tool_calls
                #    source_of_tool_calls = "kwargs"
                #    logger.info("Используются tool_calls, извлеченные из ai_message.additional_kwargs.")
                pass

        if not final_tool_calls_list:  # Если после всех проверок tool_calls_list пуст
            logger.info(
                "Router node: LLM не выбрала инструмент или не предоставила корректные tool_calls."
            )

            # Проверяем, не является ли это случаем, когда LLM сказала "tool_calls", но не дала их,
            # или дала их в некорректном формате, который не был распарсен.
            # ai_message - это оригинальный ответ от llm_with_tools.invoke(messages)
            if (
                hasattr(ai_message, "response_metadata")
                and ai_message.response_metadata.get("finish_reason") == "tool_calls"
                and (not hasattr(ai_message, "tool_calls") or not ai_message.tool_calls)
            ):

                logger.warning(
                    f"LLM-роутер ({ai_message.id if hasattr(ai_message, 'id') else 'N/A'}) "
                    f"вернул finish_reason='tool_calls', но поле tool_calls пустое или отсутствует. "
                    f"Это некорректное состояние. response_metadata: {ai_message.response_metadata}. "
                    f"Не добавляем этот AIMessage в историю. "
                    f"generate_response будет работать с предыдущей историей."
                )
                # Возвращаем исходные сообщения, без этого пустого/некорректного AIMessage
                return {
                    "next_node": "generate_response",
                    "messages": messages,  # НЕ messages + [ai_message]
                }
            else:
                # LLM не выбрала инструмент и finish_reason не 'tool_calls' (например, 'stop'),
                # ИЛИ tool_calls были, но не распарсились (хотя tool_calls_list был бы не пуст тогда).
                # Этот блок для случая, когда LLM закончила с 'stop' и есть ai_message.content.
                logger.info(
                    f"LLM-роутер сгенерировал текстовый ответ (или finish_reason не 'tool_calls'): '{ai_message.content}'"
                )
                return {
                    "next_node": "generate_response",
                    "messages": messages
                    + [ai_message],  # Добавляем ai_message с текстовым ответом
                }

        # --- ОБНОВЛЕНИЕ ЛОГИКИ ДОБАВЛЕНИЯ AIMESSAGE В ИСТОРИЮ ---
        message_to_add_to_history = ai_message  # По умолчанию оригинальное сообщение

        if source_of_tool_calls == "content" and final_tool_calls_list:
            logger.info(
                "Создание нового AIMessage, так как tool_calls были извлечены из content."
            )
            # Мы должны создать новый AIMessage, у которого поле tool_calls будет заполнено.
            # Копируем важные поля из оригинального ai_message.
            # content оставляем пустым или как есть, в зависимости от того, что правильнее.
            # Если модель вернула JSON в content, и это был ЕДИНСТВЕННЫЙ ее ответ, то content можно обнулить.
            # Если она что-то еще сказала текстом, то content оригинального ai_message может быть важен.
            # Судя по логам, content был ТОЛЬКО JSON вызова.
            new_ai_message_with_tool_calls = AIMessage(
                content="",  # Очищаем content, так как вся суть была в tool_calls
                tool_calls=final_tool_calls_list,  # Используем извлеченные tool_calls
                id=ai_message.id,  # Сохраняем оригинальный ID
                response_metadata=ai_message.response_metadata,  # Сохраняем метаданные
                # name можно не указывать, Langchain справится
            )
            message_to_add_to_history = new_ai_message_with_tool_calls
            logger.info(f"Новый AIMessage для истории: {message_to_add_to_history}")

        updated_messages = messages + [message_to_add_to_history]

        tool_call = final_tool_calls_list[0]
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
    messages = state.get("messages", [])
    if not messages:
        logger.warning("Нет сообщений для генерации ответа.")
        return {"messages": [AIMessage(content="Нет входных данных для ответа.")]}

    # Собираем только контент из BaseMessage для системного промпта,
    # а сами сообщения передаем как есть.
    # messages_for_llm = [SystemMessage(content=SYSTEM_PROMPT_RESPONSE_GENERATOR)] + messages
    # Убираем добавление системного промпта здесь, так как он может быть уже добавлен
    # или его формат может конфликтовать с ожиданиями модели при tool use.
    # Лучше управлять системным промптом на уровне конфигурации графа или начального состояния.
    messages_for_llm = list(messages)  # Копируем, чтобы не изменять оригинальный state

    # Валидация последовательности сообщений перед отправкой в LLM
    # (этот код можно вынести в отдельную утилиту, если он будет использоваться еще где-то)
    valid_sequence = True
    if not messages_for_llm:
        valid_sequence = False  # Пустая последовательность невалидна для генерации
    else:
        # Убедимся, что первое сообщение не ToolMessage или AIMessage без tool_calls
        first_msg = messages_for_llm[0]
        if isinstance(first_msg, ToolMessage) or (
            isinstance(first_msg, AIMessage)
            and not first_msg.tool_calls
            and first_msg.content == ""
        ):
            # Это не совсем корректно, но для простоты пока так.
            # В идеале, первая AIMessage в истории не должна быть пустой с tool_calls.
            logger.warning(
                f"Первое сообщение в истории для LLM имеет нежелательный тип: {type(first_msg)}"
            )
            # valid_sequence = False # Решим, насколько это критично

    # ... (существующий код валидации ToolMessage и AIMessage с tool_calls) ...
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

    # Получаем описания инструментов для передачи в LLM
    # Это важно, если LLM ожидает их при генерации ответа после ToolMessage
    tools_for_llm = [
        tool.name  # Используем только имена или можно передавать схемы, если LLM их поддерживает
        for tool_name, tool in tool_registry.items()
    ]
    # LangChain обычно ожидает формат tool.format_tool_for_openai() или аналогичный.
    # ChatOpenAI может сам позаботиться о правильном формате, если передать tool_registry
    # Однако, для простого вызова invoke/ainvoke может потребоваться явная передача `tools`.

    # Проверим, есть ли в истории ToolMessage.
    # has_tool_message = any(isinstance(msg, ToolMessage) for msg in messages_for_llm) # Это было для предыдущей логики

    try:
        # --- ИЗМЕНЕНИЕ: Убираем явную передачу tools ---
        # Если узел response_generator_node вызывается, его задача - сгенерировать текстовый ответ.
        # Передача списка инструментов здесь может путать LLM, заставляя ее пытаться снова вызвать инструмент.
        # Предполагается, что если LLM должна была вызвать инструмент, это сделал бы router_node.
        # if has_tool_message:
        #     logger.info(
        #         f"Передаем LLM список инструментов (УБРАНО ДЛЯ ТЕСТА): {[tool.name for tool in tool_registry.values()]}"
        #     )
        #     formatted_tools = []
        #     for tool_name, tool_instance in tool_registry.items():
        #         formatted_tools.append(
        #             {
        #                 "type": "function",
        #                 "function": {
        #                     "name": tool_instance.name,
        #                     "description": tool_instance.description,
        #                     "parameters": (
        #                         tool_instance.args_schema.schema()
        #                         if tool_instance.args_schema
        #                         else {}
        #                     ),
        #                 },
        #             }
        #         )
        #     logger.info(f"Явно передаем tools в ainvoke (УБРАНО ДЛЯ ТЕСТА): {formatted_tools}")
        #     response_ai_message: AIMessage = await llm.ainvoke(
        #         messages_for_llm, tools=formatted_tools  # ЯВНАЯ ПЕРЕДАЧА ИНСТРУМЕНТОВ - УБРАНО
        #     )
        # else:
        #     # Если ToolMessage в истории нет, вызываем как обычно
        logger.info(
            "Вызов LLM в response_generator_node для генерации текстового ответа (без явной передачи tools)."
        )
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
    # Используем новые имена переменных из .env согласно скриншоту
    proxy_base_url = os.getenv("base_url")
    proxy_api_key = os.getenv("api_key")

    # Переменные для YandexGPT (на случай, если LiteLLM их требует или для будущей прямой интеграции)
    # yc_folder_id = os.getenv("YC_FOLDER_ID")
    # yc_api_key = os.getenv("YC_API_KEY")

    llm = None

    if proxy_base_url:  # Проверяем новую переменную proxy_base_url
        logger.info(
            f"Обнаружен base_url (прокси): {proxy_base_url}. Используем ChatOpenAI через этот прокси."
        )
        if not llm_model_name:
            logger.error(
                "LLM_MODEL не указан, но используется прокси. Укажите модель (например, 'yandex/yandexgpt/rc' или 'gpt-3.5-turbo')."
            )
            raise ValueError("LLM_MODEL должен быть указан при использовании прокси.")
        if not proxy_api_key:  # Проверяем новую переменную proxy_api_key
            logger.warning(
                "api_key (ключ для прокси) не найден в .env, но используется прокси. "
                "Для некоторых прокси (включая LiteLLM с некоторыми моделями) ключ может быть обязательным "
                "или использоваться для маппинга. Устанавливаю 'EMPTY' по умолчанию."
            )
            proxy_api_key = "EMPTY"  # LiteLLM может работать с фиктивным ключом

        try:
            llm = ChatOpenAI(
                model=llm_model_name,  # Например, "yandex/yandexgpt/rc" или "gpt-3.5-turbo"
                temperature=0.1,  # Пример
                openai_api_base=proxy_base_url,  # Используем proxy_base_url
                openai_api_key=proxy_api_key,  # Используем proxy_api_key
            )
            logger.info(
                f"ChatOpenAI успешно инициализирован для работы с моделью '{llm_model_name}' через прокси {proxy_base_url}."
            )
        except Exception as e:
            logger.error(
                f"Ошибка при инициализации ChatOpenAI через прокси: {e}",
                exc_info=True,
            )
            raise ValueError(f"Не удалось настроить ChatOpenAI через прокси: {e}")

    else:
        # Этот блок теперь практически не должен использоваться, если всегда есть LiteLLM.
        # Оставлен для обратной совместимости или если LiteLLM не используется.
        logger.warning(
            "base_url (прокси) не указан. Попытка настроить LLM напрямую (OpenAI или YandexGPT)."
        )
        if llm_model_name and llm_model_name.startswith("yandex/") and YANDEX_VERFÜGBAR:
            logger.error(
                "Прямая инициализация YandexGPT в этом потоке больше не поддерживается. "
                "Пожалуйста, используйте LiteLLM прокси и укажите base_url."
            )
            # yc_folder_id = os.getenv("YC_FOLDER_ID")
            # yc_api_key = os.getenv("YC_API_KEY")
            # model_path_segment = llm_model_name.split("/", 1)[1]
            # model_uri = f"gpt://{yc_folder_id}/{model_path_segment.replace('/', '-')}/latest"
            # if not yc_folder_id or not yc_api_key:
            #     logger.error("YC_FOLDER_ID или YC_API_KEY не установлены для прямой работы с YandexGPT.")
            #     raise ValueError("Недостаточно данных для прямой инициализации YandexGPT.")
            # try:
            #     llm = YandexGPT(api_key=yc_api_key, folder_id=yc_folder_id, model_uri=model_uri, temperature=0.1, max_tokens=1500)
            #     logger.info("YandexGPT (прямое подключение) успешно инициализирован.")
            # except Exception as e:
            #     logger.error(f"Ошибка при прямой инициализации YandexGPT: {e}", exc_info=True)
            #     raise ValueError(f"Не удалось настроить YandexGPT напрямую: {e}")
            raise NotImplementedError(
                "Прямая инициализация YandexGPT удалена. Используйте LiteLLM прокси."
            )

        elif (
            llm_model_name
        ):  # Подразумевается OpenAI или другая OpenAI-совместимая модель напрямую
            openai_model_to_use = (
                llm_model_name  # Используем LLM_MODEL как имя модели OpenAI
            )
            logger.info(
                f"Используется модель OpenAI (прямое подключение): {openai_model_to_use}"
            )
            if not proxy_api_key:  # Проверяем proxy_api_key и здесь на всякий случай
                logger.error(
                    "api_key не найден. Прямое подключение к OpenAI невозможно (если это предполагалось)."
                )
                raise ValueError("api_key не указан для прямого подключения к OpenAI.")
            try:
                llm = ChatOpenAI(
                    model=openai_model_to_use,
                    temperature=0,
                    openai_api_key=proxy_api_key,  # Используем proxy_api_key
                )
                logger.info("ChatOpenAI (прямое подключение) успешно инициализирован.")
            except Exception as e:
                logger.error(
                    f"Ошибка при прямой инициализации ChatOpenAI: {e}", exc_info=True
                )
                raise ValueError(f"Не удалось настроить ChatOpenAI напрямую: {e}")
        else:
            logger.error(
                "Не удалось определить, какую LLM инициализировать. Проверьте LLM_MODEL и base_url."
            )
            raise ValueError("Недостаточно конфигурации для инициализации LLM.")

    if llm is None:
        # Эта проверка должна быть избыточной, если предыдущая логика корректно бросает ошибки
        logger.critical("КРИТИЧЕСКАЯ ОШИБКА: LLM не был инициализирован!")
        raise ValueError(
            "Не удалось настроить ни одну LLM. Проверьте переменные окружения и логи."
        )

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
            "Внимание: Ни один инструмент не был успешно инициализирован! "
            "Агент будет работать без инструментов (кроме режима теста Tavily)."
        )  # В этом случае LLM все еще может быть инициализирована
    elif not TEST_TAVILY_ONLY_MODE:
        logger.info(f"Зарегистрированные инструменты: {list(tool_registry.keys())}")

    app = build_graph(llm)
    logger.info("Агент успешно настроен и граф скомпилирован.")
    return app
