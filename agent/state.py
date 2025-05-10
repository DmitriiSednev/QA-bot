from typing import TypedDict, Annotated, Sequence, Optional, Dict, Any
import operator

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """Определяет состояние графа LangGraph для QA-бота.

    Attributes:
        messages: История сообщений диалога. Использует operator.add для накопления.
        # Дополнительные поля могут быть добавлены позже:
        # - context: Дополнительный контекст из RAG или инструментов.
        # - plan: План действий, сгенерированный планировщиком.
        # - tool_result: Результат последнего вызванного инструмента.
        # - next_node: Указание на следующий узел для условных переходов.
    """

    messages: Annotated[Sequence[BaseMessage], operator.add]

    # Поля для управления потоком и инструментами
    next_node: Optional[str] = None
    tool_to_call: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_result: Optional[Any] = None  # Может быть строка или другой объект
    faq_search_result: Optional[str] = None  # Результат поиска по FAQ
    current_tool_call_id: Optional[str] = None
    tool_name_executed: Optional[str] = None


def response_generator_node(state: AgentState, llm):
    # ... существующий код ...
    context = ""
    if state.get("tool_result"):
        context += f"Контекст из документации Yandex Cloud:\n{state['tool_result']}\n"
    # ... остальной prompt ...
