import logging
from typing import Type, List, Dict, Any, Optional
from datetime import datetime, timedelta

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

from database import crud, connection, models
from .base_search_tool import BaseSearchTool, BaseSearchInput

logger = logging.getLogger(__name__)

class ChatHistorySearchInput(BaseSearchInput):
    """Схема входных данных для ChatHistorySearchTool."""
    chat_id: int = Field(description="ID чата для поиска")
    time_window: int = Field(default=7, description="За сколько последних дней искать")

class ChatHistorySearchTool(BaseSearchTool):
    """Инструмент для поиска похожих вопросов в истории чата."""

    name: str = "search_chat_history"
    description: str = (
        "Используй этот инструмент для поиска похожих вопросов в истории чата. "
        "Это поможет найти ранее заданные похожие вопросы и ответы на них. "
        "Особенно полезно, когда текущий вопрос похож на те, что уже задавались ранее."
    )
    args_schema: Type[BaseModel] = ChatHistorySearchInput

    def _run(
        self,
        query: str,
        chat_id: int,
        time_window: int = 7,
        limit: int = 5,
        min_similarity: float = 0.7,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Ищет похожие вопросы в истории чата."""
        logger.info(f"Поиск похожих вопросов для: {query}")

        # Проверяем кэш
        cache_key = f"{query}_{chat_id}_{time_window}_{limit}_{min_similarity}"
        cached_results = self._get_cached_results(query, limit, min_similarity)
        if cached_results:
            return self._format_results(cached_results)

        try:
            with connection.get_db_session() as db:
                if not db:
                    return "Ошибка: Не удалось подключиться к базе данных."

                results = crud.search_chat_history_optimized(
                    db,
                    query,
                    chat_id=chat_id,
                    limit=limit,
                    min_similarity=min_similarity,
                    time_window=time_window
                )
                
                # Преобразуем результаты в формат для кэширования
                formatted_results = [
                    {
                        'title': entry.message_text,
                        'content': entry.response_text or "Нет ответа",
                        'similarity': entry.similarity if hasattr(entry, 'similarity') else None,
                        'date': entry.created_at.strftime("%Y-%m-%d %H:%M")
                    }
                    for entry in results
                ]

                # Кэшируем результаты
                self._cache_results(query, limit, min_similarity, formatted_results)

                return self._format_results(formatted_results)

        except Exception as e:
            logger.error(f"Ошибка при поиске в истории чата: {e}", exc_info=True)
            return f"Произошла ошибка при поиске в истории чата: {e}" 