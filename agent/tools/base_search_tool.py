from typing import Type, Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
import logging
from functools import lru_cache
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class BaseSearchInput(BaseModel):
    """Базовый класс для входных данных поиска."""
    query: str = Field(description="Поисковый запрос")
    limit: int = Field(default=5, description="Максимальное количество результатов")
    min_similarity: float = Field(default=0.7, description="Минимальное сходство для результатов")

class BaseSearchTool(BaseTool):
    """Базовый класс для всех инструментов поиска."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = {}
        self._cache_ttl = timedelta(minutes=5)
    
    @lru_cache(maxsize=1000)
    def _get_cached_results(self, query: str, limit: int, min_similarity: float) -> Optional[List[Dict[str, Any]]]:
        """Получает результаты из кэша."""
        cache_key = f"{query}_{limit}_{min_similarity}"
        if cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            if datetime.now() - cache_entry['timestamp'] < self._cache_ttl:
                return cache_entry['results']
            del self._cache[cache_key]
        return None
    
    def _cache_results(self, query: str, limit: int, min_similarity: float, results: List[Dict[str, Any]]):
        """Сохраняет результаты в кэш."""
        cache_key = f"{query}_{limit}_{min_similarity}"
        self._cache[cache_key] = {
            'results': results,
            'timestamp': datetime.now()
        }
    
    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """Форматирует результаты поиска."""
        if not results:
            return "Ничего не найдено."
        
        response = f"Найдено {len(results)} результатов:\n\n"
        for i, result in enumerate(results, 1):
            response += f"{i}. {result.get('title', 'Без заголовка')}\n"
            if 'content' in result:
                response += f"   {result['content'][:200]}...\n"
            if 'url' in result:
                response += f"   Ссылка: {result['url']}\n"
            response += "\n"
        
        return response.strip()
    
    def _run(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.7,
        run_manager: CallbackManagerForToolRun | None = None
    ) -> str:
        """Базовый метод поиска."""
        raise NotImplementedError("Метод _run должен быть реализован в дочерних классах") 