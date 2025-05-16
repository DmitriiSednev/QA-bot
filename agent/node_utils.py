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

# --- Настройка логгера ---
logger = logging.getLogger(__name__)

# --- Глобальная переменная для LLM (устанавливается в llm_setup.py) ---
llm_instance: Optional[ChatOpenAI] = None

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
        r'(tavily_yandexcloud_search).*?query[\":= ]+["\']?([^"\'}\n]+)',
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