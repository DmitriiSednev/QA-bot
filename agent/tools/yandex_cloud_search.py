import logging
from typing import Type, List, Dict, Any, Optional
from datetime import datetime, timedelta

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
import requests
import os
from bs4 import BeautifulSoup

from .base_search_tool import BaseSearchTool, BaseSearchInput

logger = logging.getLogger(__name__)

class YandexCloudSearchInput(BaseSearchInput):
    """Схема входных данных для YandexCloudSearchTool."""
    pass

class SearchResult(BaseModel):
    """Модель для хранения результата поиска."""
    title: str
    url: str
    snippet: str
    source_content: str = ""

class YandexCloudSearchTool(BaseSearchTool):
    """Инструмент для поиска в документации Yandex Cloud."""

    name: str = "yandex_cloud_docs_search"
    description: str = (
        "Используй этот инструмент для поиска информации ИСКЛЮЧИТЕЛЬНО в официальной документации Yandex Cloud. "
        "Использовать ТОЛЬКО для вопросов о Yandex Cloud, и ТОЛЬКО ЕСЛИ поиск по FAQ не дал ответа."
    )
    args_schema: Type[BaseModel] = YandexCloudSearchInput

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("API_YC")
        if not self.api_key:
            raise ValueError("API_YC environment variable is not set")

    def _extract_content(self, url: str) -> str:
        """Извлекает основной контент со страницы."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Удаляем ненужные элементы
            for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
                tag.decompose()
            
            # Получаем основной контент
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
            if main_content:
                return main_content.get_text(strip=True, separator=' ')
            return soup.get_text(strip=True, separator=' ')
        except Exception as e:
            logger.error(f"Ошибка при извлечении контента с {url}: {e}")
            return ""

    def _run(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.7,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Ищет информацию в документации Yandex Cloud."""
        logger.info(f"Поиск в документации Yandex Cloud: {query}")

        # Проверяем кэш
        cache_key = f"{query}_{limit}_{min_similarity}"
        cached_results = self._get_cached_results(query, limit, min_similarity)
        if cached_results:
            return self._format_results(cached_results)

        try:
            # Здесь будет реализация поиска через API Yandex Cloud
            # Пока возвращаем заглушку
            results = [
                {
                    'title': 'Документация Yandex Cloud',
                    'content': 'Здесь будет результат поиска',
                    'url': 'https://cloud.yandex.ru/docs'
                }
            ]

            # Кэшируем результаты
            self._cache_results(query, limit, min_similarity, results)

            return self._format_results(results)

        except Exception as e:
            logger.error(f"Ошибка при поиске в документации Yandex Cloud: {e}", exc_info=True)
            return f"Произошла ошибка при поиске в документации Yandex Cloud: {e}"

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """Форматирует результаты поиска для использования в LLM."""
        formatted_results = "\n\n".join([
            f"Источник: {result['title']}\nURL: {result['url']}\n"
            f"Сниппет: {result['content']}\n"
            f"Контент: {result['source_content'][:1000]}..."  # Ограничиваем размер для LLM
            for result in results
        ])
        return formatted_results 