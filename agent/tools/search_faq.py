import logging
from typing import Type, List, Dict, Any

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

# Импортируем CRUD операции и функцию получения сессии
from database import connection, models # models нужен для аннотации типа
from database.crud_faq import search_faq_entries
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
        """Ищет записи в FAQ и возвращает найденные результаты, используя Redis кеш."""
        logger.info(f"Запуск SearchFAQTool с запросом: '{query}', limit={limit}, min_similarity={min_similarity}")

        # Генерируем ключ кеша
        cache_key = self._get_cache_key(query=query, limit=limit, min_similarity=min_similarity)

        # Проверяем кеш Redis
        cached_results = self._get_cached_results(cache_key)
        if cached_results is not None: # Проверяем именно на None, т.к. пустой список - валидный кешированный результат
            logger.info(f"Результаты для ключа '{cache_key}' найдены в кеше Redis.")
            return self._format_results(cached_results)
        else:
            logger.info(f"Результаты для ключа '{cache_key}' НЕ найдены в кеше Redis. Выполняем поиск.")

        try:
            with connection.get_db_session() as db:
                if not db:
                    return "Ошибка: Не удалось получить сессию базы данных."

                # Используем импортированную функцию напрямую
                search_results: List[models.FAQEntry] = search_faq_entries(
                    db, query, limit=limit, min_similarity=min_similarity
                )

                # Преобразуем результаты в формат для кэширования (список словарей)
                # Добавляем similarity, если оно есть (зависит от реализации search_faq_entries)
                formatted_results_for_cache: List[Dict[str, Any]] = [
                    {
                        'id': entry.id,
                        'question': entry.question,
                        'answer': entry.answer,
                        'created_at': entry.created_at.isoformat() if entry.created_at else None,
                        # Добавляем 'similarity', если оно было возвращено функцией поиска
                        # 'similarity': getattr(entry, 'similarity', None) # Пример
                    }
                    for entry in search_results
                ]
                
                # Адаптируем форматирование для вывода пользователю
                formatted_results_for_output: List[Dict[str, Any]] = [
                     {
                        'title': entry.question, # Используем 'title' для форматирования
                        'content': entry.answer, # Используем 'content'
                        # 'similarity': getattr(entry, 'similarity', None) # Можно добавить для вывода
                    }
                    for entry in search_results
                ]

                # Кэшируем результаты в Redis (TTL по умолчанию 1 час)
                self._cache_results(cache_key, formatted_results_for_cache)

                return self._format_results(formatted_results_for_output)

        except Exception as e:
            logger.error(
                f"Ошибка в SearchFAQTool при запросе '{query}': {e}", exc_info=True
            )
            return f"Произошла ошибка при поиске в FAQ: {e}"
