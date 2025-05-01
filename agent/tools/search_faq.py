import logging
from typing import Type, List, Dict, Any

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

# Импортируем CRUD операции и функцию получения сессии
from database import crud, connection, models  # models нужен для аннотации типа
from .base_search_tool import BaseSearchTool, BaseSearchInput

logger = logging.getLogger(__name__)


class SearchFAQInput(BaseSearchInput):
    """Схема входных данных для SearchFAQTool."""
    pass


class SearchFAQTool(BaseSearchTool):
    """Инструмент для поиска релевантных записей в базе знаний FAQ."""

    name: str = "search_faq"
    description: str = (
        "Используй этот инструмент для поиска ответов на вопросы пользователя "
        "во внутренней базе знаний (FAQ). Особенно полезен для вопросов, "
        "касающихся специфики проекта или ранее обсуждавшихся тем. "
        "Входными данными должен быть поисковый запрос (вопрос пользователя)."
    )
    args_schema: Type[BaseModel] = SearchFAQInput

    def _run(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.7,
        run_manager: CallbackManagerForToolRun | None = None
    ) -> str:
        """Ищет записи в FAQ и возвращает найденные результаты."""
        logger.info(f"Запуск SearchFAQTool с запросом: {query}")

        # Проверяем кэш
        cached_results = self._get_cached_results(query, limit, min_similarity)
        if cached_results:
            return self._format_results(cached_results)

        try:
            with connection.get_db_session() as db:
                if not db:
                    return "Ошибка: Не удалось получить сессию базы данных."

                # Используем оптимизированный поиск
                search_results: List[models.FAQEntry] = crud.search_faq_entries(
                    db, query, limit=limit
                )

                # Преобразуем результаты в формат для кэширования
                formatted_results = [
                    {
                        'title': entry.question,
                        'content': entry.answer,
                        'similarity': entry.similarity if hasattr(entry, 'similarity') else None
                    }
                    for entry in search_results
                ]

                # Кэшируем результаты
                self._cache_results(query, limit, min_similarity, formatted_results)

                return self._format_results(formatted_results)

        except Exception as e:
            logger.error(
                f"Ошибка в SearchFAQTool при запросе '{query}': {e}", exc_info=True
            )
            return f"Произошла ошибка при поиске в FAQ: {e}"
