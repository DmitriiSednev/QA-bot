from typing import Type, Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
import logging
# Убираем lru_cache и datetime, они теперь не нужны в базовом классе

# Импортируем функции кеширования Redis
from database.redis_cache import get_cache, set_cache

logger = logging.getLogger(__name__)

class BaseSearchInput(BaseModel):
    """Базовый класс для входных данных поиска."""
    query: str = Field(description="Поисковый запрос")
    limit: int = Field(default=5, description="Максимальное количество результатов")
    min_similarity: float = Field(default=0.7, description="Минимальное сходство для результатов")

class BaseSearchTool(BaseTool):
    """Базовый класс для всех инструментов поиска с кешированием Redis."""
    
    # Убираем __init__ с self._cache и self._cache_ttl

    def _get_cache_key(self, *args, **kwargs) -> str:
        """Генерирует ключ кеша на основе имени инструмента и его аргументов."""
        # Используем имя класса инструмента для уникальности префикса
        prefix = self.name or self.__class__.__name__
        # Формируем строку из аргументов для ключа
        # Преобразуем все аргументы в строку, сортируем для консистентности
        args_str = "_".join(map(str, args))
        kwargs_str = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        # Не используем query напрямую в ключе, если он слишком длинный, 
        # лучше использовать хэш, но для простоты пока оставим так.
        # Убедимся, что строка ключа не слишком длинная и содержит только безопасные символы.
        key_parts = [prefix, args_str, kwargs_str]
        raw_key = ":".join(filter(None, key_parts)) # Соединяем части через :
        # Простая очистка ключа (можно заменить на хэширование для надежности)
        safe_key = "".join(c if c.isalnum() or c in ':_-' else '_' for c in raw_key)[:250] # Ограничим длину
        return safe_key

    def _get_cached_results(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Получает результаты из кеша Redis."""
        return get_cache(cache_key)

    def _cache_results(self, cache_key: str, results: List[Dict[str, Any]], ttl_seconds: int = 3600):
        """Сохраняет результаты в кеш Redis."""
        set_cache(cache_key, results, ttl_seconds=ttl_seconds)

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """Форматирует результаты поиска."""
        if not results:
            return "Ничего не найдено."
        
        response = f"Найдено {len(results)} результатов:\n\n"
        for i, result in enumerate(results, 1):
            response += f"{i}. {result.get('title', 'Без заголовка')}\n"
            if 'content' in result:
                response += f"   {result['content'][:200]}...\n"
            # Добавляем доп. поля, если они есть (например, similarity, date)
            if 'similarity' in result and result['similarity'] is not None:
                response += f"   Сходство: {result['similarity']: .3f}\n"
            if 'date' in result:
                response += f"   Дата: {result['date']}\n"
            if 'url' in result:
                 response += f"   Ссылка: {result['url']}\n"
            response += "\n"
        
        return response.strip()
    
    def _run(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.7,
        run_manager: CallbackManagerForToolRun | None = None,
        **kwargs: Any # Принимаем доп. аргументы для ключа кеша
    ) -> str:
        """Базовый метод поиска. Должен быть переопределен.
           Реализация кеширования перенесена в вызывающие методы инструментов.
        """
        raise NotImplementedError("Метод _run должен быть реализован в дочерних классах") 