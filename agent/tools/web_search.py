import logging
from typing import Type, Optional
from langchain_core.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.callbacks import CallbackManagerForToolRun
from tavily import TavilyClient
import os

logger = logging.getLogger(__name__)


class WebSearchInput(BaseModel):
    """Input for the WebSearchTool."""

    query: str = Field(description="Поисковый запрос для поиска в документации Yandex Cloud.")


class WebSearchTool(BaseTool):
    """Инструмент для поиска информации в документации Yandex Cloud."""

    name: str = "yandex_cloud_docs_search"
    description: str = (
        "Используй этот инструмент ТОЛЬКО для поиска информации в документации Yandex Cloud (cloud.yandex.ru). "
        "НЕ ИСПОЛЬЗУЙ для общих вопросов или поиска вне Yandex Cloud. "
        "Входными данными должен быть поисковый запрос по Yandex Cloud."
    )
    args_schema: Type[BaseModel] = WebSearchInput
    max_results: int = 5

    tavily_client: Optional[TavilyClient] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if tavily_api_key:
            self.tavily_client = TavilyClient(api_key=tavily_api_key)
        else:
            logger.warning("TAVILY_API_KEY не найден в переменных окружения. Веб-поиск может не работать.")

    def _run(
        self, query: str, run_manager: CallbackManagerForToolRun | None = None
    ) -> str:
        """Использует инструмент и возвращает результат поиска по документации Yandex Cloud."""
        search_query = f"site:cloud.yandex.ru {query}"
        logger.info(f"Запуск {self.name} с модифицированным запросом: {search_query}")

        if not self.tavily_client:
            return "Ошибка: Клиент веб-поиска (Tavily) не инициализирован. Проверьте TAVILY_API_KEY."
        try:
            results = self.tavily_client.search(query=search_query, search_depth="basic", max_results=self.max_results)
            search_results = results.get('results', [])

            if not search_results:
                return f"В документации Yandex Cloud по запросу '{query}' ничего не найдено."

            result_string = f"Результаты поиска по документации Yandex Cloud ({query}):\n\n"
            for i, res in enumerate(search_results, 1):
                result_string += f"{i}. {res.get('title', 'Нет заголовка')}\n"
                result_string += f"   URL: {res.get('url', 'Нет URL')}\n"
                result_string += f"   Содержимое: {res.get('content', 'Нет описания')[:500]}...\n\n"
            return result_string

        except Exception as e:
            logger.error(
                f"Ошибка в {self.name} при запросе '{search_query}': {e}", exc_info=True
            )
            return f"Произошла ошибка при выполнении поиска по документации Yandex Cloud: {e}"
