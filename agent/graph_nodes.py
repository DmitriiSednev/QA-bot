import logging
import json
from typing import Annotated, Optional, List, Dict, Any, Sequence, Tuple, Union, Literal
import operator
import os

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
    # Получаем LLM, установленную через set_node_llm
    global llm
    if not llm:
        logger.error("Router Node: LLM не инициализирована!")
        return {
            "messages": add_messages(
                state["messages"], [AIMessage(content="Ошибка: LLM не настроена.")]
            )
        }

    # --- Новый подход: Ручной выбор инструмента через промпт ---

    # 1. Формируем список инструментов для промпта
    available_tools_prompt = "\n".join(
        f"- `{tool.name}`: {tool.description}" for tool in tools_list
    )

    # 2. Создаем системный промпт для выбора инструмента
    system_prompt_text = f"""Ты - ИИ-маршрутизатор. Твоя задача - проанализировать последний запрос пользователя и историю диалога, а затем решить, нужно ли вызывать один из доступных инструментов для получения дополнительной информации или выполнения действия. 

Доступные инструменты:
{available_tools_prompt}

**Правила выбора:**
- Если запрос касается **Yandex Cloud**, его сервисов, API, CLI и т.д., **В ПЕРВУЮ ОЧЕРЕДЬ** выбери инструмент `search_yandex_documentation`.
- Если нужен поиск по внутренней базе знаний (FAQ), используй `search_faq`.
- Если нужно выполнить операцию с FAQ (добавить, обновить, удалить), используй `add_faq`, `update_faq`, `delete_faq`.
- Если нужно поискать в истории прошлых сообщений, используй `search_chat_history`.
- Если инструмент не нужен или информации достаточно для ответа, выбери "no_tool".

**Формат ответа:**
Верни ТОЛЬКО JSON объект со следующей структурой:
{{ "tool_name": "<название_инструмента_или_no_tool>", "tool_input": {{ <аргументы_инструмента_в_формате_ключ_значение> }} }}

Если инструмент не нужен, верни: {{ "tool_name": "no_tool", "tool_input": {{}} }}
Если вызываешь инструмент, укажи его точное имя в `tool_name` и все необходимые аргументы в `tool_input`. Например, для `search_faq` нужен аргумент `query`: {{ "tool_name": "search_faq", "tool_input": {{"query": "текст запроса пользователя"}} }}
"""

    # 3. Формируем промпт для LLM
    router_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_text),
            MessagesPlaceholder(variable_name="messages", optional=True),
            # Добавляем явное указание на последний запрос
            (
                "human",
                "Проанализируй последний запрос и реши, какой инструмент вызвать или ответь 'no_tool' в JSON формате.",
            ),
        ]
    )

    chain = router_prompt | llm  # Используем LLM без bind_tools

    try:
        MAX_ROUTER_MESSAGES = 10
        routing_messages = state["messages"][-MAX_ROUTER_MESSAGES:]
        routing_messages = [
            m for m in routing_messages if m.type != "system"
        ]  # Убираем системные

        logger.debug(
            f"Router Node: Отправка сообщений в LLM для выбора инструмента: {routing_messages}"
        )
        ai_response = await chain.ainvoke({"messages": routing_messages})
        logger.debug(f"Router Node: Получен ответ от LLM: {ai_response.content}")

        # 4. Парсим JSON ответ от LLM
        try:
            # Убираем возможные ```json ``` обертки
            cleaned_content = ai_response.content.strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content[7:]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3]

            decision = json.loads(cleaned_content)
            tool_name = decision.get("tool_name")
            tool_input = decision.get("tool_input", {})  # Если нет, пустой dict

            if not isinstance(tool_input, dict):
                logger.warning(
                    f"Router Node: tool_input в ответе LLM не является словарем: {tool_input}. Использую {{}}."
                )
                tool_input = {}

        except json.JSONDecodeError:
            logger.error(
                f"Router Node: Не удалось распарсить JSON от LLM: {ai_response.content}"
            )
            # Если не удалось распарсить, считаем, что инструмент не нужен
            tool_name = "no_tool"
            tool_input = {}
        except Exception as parse_err:
            logger.error(
                f"Router Node: Ошибка при обработке ответа LLM: {parse_err}",
                exc_info=True,
            )
            tool_name = "no_tool"
            tool_input = {}

        # 5. Возвращаем результат
        if tool_name == "no_tool" or tool_name is None:
            logger.info("Router Node: LLM решила не вызывать инструмент.")
            # Мы не можем просто передать управление response_generator,
            # так как граф ожидает либо tool_calls, либо просто AIMessage.
            # Добавим оригинальный ответ LLM (который был JSON или ошибкой)
            # и укажем, что инструмент не нужен.
            # Генератор ответа должен будет сам понять, что делать.
            return {
                "messages": add_messages(state["messages"], [ai_response]),
                # НЕ устанавливаем tool_to_call, tool_input, current_tool_call_id
            }
        else:
            logger.info(
                f"Router Node: LLM выбрала инструмент: '{tool_name}' с аргументами: {tool_input}"
            )
            # Здесь нам нужно создать фейковый AIMessage с tool_calls,
            # чтобы узел tool_executor мог его обработать.
            # Генерируем фейковый tool_call_id.
            tool_call_id = f"tool_call_{tool_name}_{os.urandom(4).hex()}"

            fake_ai_message = AIMessage(
                content=f"(Решение маршрутизатора: вызвать инструмент {tool_name})",  # Контент для отладки
                tool_calls=[
                    {
                        "id": tool_call_id,
                        "name": tool_name,
                        "args": tool_input,
                    }
                ],
            )

            # Возвращаем фейковое сообщение и данные для исполнителя
            # Важно: добавляем fake_ai_message в history
            return {
                "messages": add_messages(state["messages"], [fake_ai_message]),
                "tool_to_call": tool_name,
                "tool_input": tool_input,
                # current_tool_call_id теперь не нужен, т.к. мы генерируем его здесь
                # tool_executor будет использовать ID из fake_ai_message
            }

    except Exception as e:
        logger.error(
            f"Router Node: Ошибка при вызове LLM для маршрутизации: {e}", exc_info=True
        )
        error_message = AIMessage(content=f"Ошибка при принятии решения: {e}")
        return {"messages": add_messages(state["messages"], [error_message])}


# --- Узел выполнения инструментов --- #
def tool_executor_node(state):
    """Выполняет вызванные инструменты."""
    logger.debug("===Tool Executor Node===")
    messages = state["messages"]
    last_message = messages[-1]

    # Ожидаем наш фейковый AIMessage
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        logger.debug(
            "Tool Executor Node: Последнее сообщение не содержит tool_calls (ожидалось от router_node)."
        )
        # Если router решил не вызывать инструмент, он не добавляет tool_calls.
        # В этом случае executor не должен ничего делать.
        # Условие перехода в графе должно обработать этот случай.
        return {}  # Ничего не делаем

    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]  # Берем ID из сообщения

        logger.info(
            f"Tool Executor Node: Вызов инструмента '{tool_name}' с ID '{tool_call_id}' и аргументами: {tool_args}"
        )

        tool = next((t for t in tools_list if t.name == tool_name), None)

        if tool is None:
            is_error = True
            content = f"Ошибка: инструмент '{tool_name}' не найден."
        else:
            try:
                # Убедимся, что tool_args это словарь
                if not isinstance(tool_args, dict):
                    logger.warning(
                        f"Tool Executor: tool_args для {tool_name} не словарь: {tool_args}. Преобразую в {{}}."
                    )
                    tool_args = {}

                # Используем invoke напрямую, т.к. _run часто ожидает распакованные аргументы
                # Если инструмент использует Pydantic модель, invoke(tool_args) должен работать
                # Если инструмент ожидает _run(self, arg1, arg2), то invoke(tool_args) МОЖЕТ вызвать ошибку
                # Проверяем, есть ли у инструмента args_schema и как он реализован
                # Если args_schema есть, invoke(tool_args) предпочтительнее
                if hasattr(tool, "args_schema") and tool.args_schema:
                    # Валидация Pydantic должна сработать внутри invoke
                    content = tool.invoke(tool_args)
                else:
                    # Если схемы нет, пытаемся вызвать _run с распаковкой словаря
                    # Это менее надежно
                    content = tool._run(**tool_args)

                # Альтернативно, можно всегда использовать invoke
                # content = tool.invoke(tool_args)

                is_error = False
            except Exception as e:
                logger.error(
                    f"Ошибка выполнения инструмента '{tool_name}': {e}", exc_info=True
                )
                content = f"Ошибка выполнения инструмента: {e}"
                is_error = True

        logger.info(
            f"Tool Executor Node: Результат '{tool_name}': {str(content)[:200]}...{'(Ошибка)' if is_error else ''}"
        )

        tool_messages.append(
            ToolMessage(
                tool_call_id=tool_call_id,
                content=str(content),  # Преобразуем результат в строку
                additional_kwargs={"is_error": is_error},
                name=tool_name,  # Добавим имя для ясности, хотя оно есть в ID
            )
        )

    # Возвращаем ТОЛЬКО новые сообщения ToolMessage
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
