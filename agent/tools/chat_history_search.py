import logging
from typing import Type, List, Dict, Any, Optional
from datetime import datetime, timedelta

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

from database import connection, models
from database.crud_chat_history import search_chat_history
from .base_search_tool import BaseSearchTool, BaseSearchInput

logger = logging.getLogger(__name__)

class ChatHistorySearchInput(BaseSearchInput):
    """Схема входных данных для ChatHistorySearchTool."""
    chat_id: Optional[int] = Field(None, description="ID чата Telegram, в котором искать. Если None, поиск по всем чатам.")
    time_window_days: Optional[int] = Field(None, description="Ограничить поиск последними N днями. Если None, поиск без ограничения по времени.")

class ChatHistorySearchTool(BaseSearchTool):
    """Инструмент для поиска релевантных сообщений в истории чатов."""

    name: str = "search_chat_history"
    description: str = (
        "Используй этот инструмент для поиска похожих вопросов или сообщений в истории прошлых диалогов. "
        "Полезен для поиска контекста или ранее данных ответов. "
        "Можно указать ID чата для поиска только в нем, или искать по всем чатам. "
        "Также можно ограничить поиск по времени (в днях)."
    )
    args_schema: Type[BaseModel] = ChatHistorySearchInput

    def _run(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.7,
        chat_id: Optional[int] = None,
        time_window_days: Optional[int] = None,
        run_manager: CallbackManagerForToolRun | None = None
    ) -> str:
        """Ищет записи в истории чатов."""
        logger.info(
            f"Запуск ChatHistorySearchTool: query='{query}', chat_id={chat_id}, "
            f"limit={limit}, min_similarity={min_similarity}, days={time_window_days}"
        )

        # Генерируем ключ кеша
        cache_key = self._get_cache_key(
            query=query,
            limit=limit,
            min_similarity=min_similarity,
            chat_id=chat_id,
            time_window_days=time_window_days,
        )

        # Проверяем кеш Redis
        cached_results = self._get_cached_results(cache_key)
        if cached_results is not None:
            logger.info(f"Результаты для ключа '{cache_key}' найдены в кеше Redis (история чата).")
            return self._format_results(cached_results)
        else:
            logger.info(f"Результаты для ключа '{cache_key}' НЕ найдены в кеше Redis (история чата).")

        try:
            with connection.get_db_session() as db:
                if not db:
                    return "Ошибка: Не удалось получить сессию базы данных."

                # Используем импортированную функцию напрямую
                search_results: List[models.ChatHistory] = search_chat_history(
                    db,
                    query,
                    chat_id=chat_id,
                    limit=limit,
                    min_similarity=min_similarity,
                    time_window_days=time_window_days,
                )

                # Преобразуем для кеша
                formatted_results_for_cache: List[Dict[str, Any]] = [
                    {
                        'id': entry.id,
                        'chat_id': entry.chat_id,
                        'user_id': entry.user_id,
                        'question': entry.message_text,
                        'answer': entry.response_text,
                        'is_answered': entry.is_answered,
                        'created_at': entry.created_at.isoformat() if entry.created_at else None,
                        # 'similarity': getattr(entry, 'similarity', None)
                    }
                    for entry in search_results
                ]
                
                # Преобразуем для вывода
                formatted_results_for_output: List[Dict[str, Any]] = [
                    {
                        'title': f"Chat {entry.chat_id}, User {entry.user_id} ({entry.created_at.strftime('%Y-%m-%d %H:%M') if entry.created_at else 'N/A'})",
                        'content': f"Q: {entry.message_text}\nA: {entry.response_text if entry.response_text else '(No recorded answer)'}",
                        # 'similarity': getattr(entry, 'similarity', None)
                    }
                    for entry in search_results
                ]

                # Кэшируем результаты в Redis
                self._cache_results(cache_key, formatted_results_for_cache)

                return self._format_results(formatted_results_for_output)

        except Exception as e:
            logger.error(
                f"Ошибка в ChatHistorySearchTool при запросе '{query}': {e}",
                exc_info=True,
            )
            return f"Произошла ошибка при поиске в истории чатов: {e}" 