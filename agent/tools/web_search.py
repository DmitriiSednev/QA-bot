import logging
from typing import Type

from duckduckgo_search import DDGS
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class WebSearchInput(BaseModel):
    """Input for the WebSearchTool."""

    query: str = Field(description="Поисковый запрос для поиска в интернете.")


class WebSearchTool(BaseTool):
    """Инструмент для поиска актуальной информации в интернете с помощью DuckDuckGo."""

    name: str = "web_search"
    description: str = (
        "Используй этот инструмент для поиска актуальной информации в интернете "
        "или для ответа на вопросы об общеизвестных фактах, текущих событиях, "
        "или когда внутренняя база знаний (FAQ) не содержит ответа. "
        "Входными данными должен быть поисковый запрос."
    )
    args_schema: Type[BaseModel] = WebSearchInput
    max_results: int = 5  # Ограничим количество результатов

    def _run(
        self, query: str, run_manager: CallbackManagerForToolRun | None = None
    ) -> str:
        """Использует инструмент и возвращает результат."""
        logger.info(f"Запуск WebSearchTool с запросом: {query}")
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))
                if not results:
                    return "По вашему запросу в интернете ничего не найдено."

                # Форматируем результат для LLM
                result_string = "Результаты веб-поиска:\n\n"
                for i, res in enumerate(results, 1):
                    result_string += f"{i}. {res.get('title', 'Нет заголовка')}\n"
                    result_string += f"   URL: {res.get('href', 'Нет URL')}\n"
                    result_string += (
                        f"   Фрагмент: {res.get('body', 'Нет описания')}\n\n"
                    )
                return result_string

        except Exception as e:
            logger.error(
                f"Ошибка в WebSearchTool при запросе '{query}': {e}", exc_info=True
            )
            return f"Произошла ошибка при выполнении веб-поиска: {e}"
