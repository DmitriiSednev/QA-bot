from typing import TypedDict, Annotated, Sequence, Optional, Dict, Any, List
import operator
from datetime import datetime

from langchain_core.messages import BaseMessage


class MessageContext(TypedDict):
    """Контекст сообщения для анализа."""
    chat_type: str  # private, group, supergroup
    is_mentioned: bool
    is_reply_to_bot: bool
    message_time: datetime
    chat_id: int
    user_id: int
    username: Optional[str]
    message_text: str
    previous_messages: List[Dict[str, Any]]  # История последних сообщений


class AgentState(TypedDict):
    """Определяет состояние графа LangGraph для QA-бота."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    context: Optional[MessageContext]
    should_respond: bool
    confidence: float
    last_response_time: Optional[datetime]
    conversation_history: List[Dict[str, Any]]
    current_tool: Optional[str]
    tool_history: List[Dict[str, Any]]
    error_count: int
    retry_count: int
    next_node: Optional[str]
    tool_to_call: Optional[str]
    tool_input: Optional[Dict[str, Any]]
    tool_result: Optional[Any]
    faq_search_result: Optional[str]
    current_tool_call_id: Optional[str]
    tool_name_executed: Optional[str]
    user_id: Optional[int]
