import logging
from typing import Type, List

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

# Импортируем CRUD операции и функцию получения сессии
from database import crud, connection, models  # models нужен для аннотации типа

logger = logging.getLogger(__name__)


class SearchFAQInput(BaseModel):
    """Схема входных данных для SearchFAQTool."""

    query: str = Field(description="Текст вопроса для поиска в базе знаний FAQ.")


class SearchFAQTool(BaseTool):
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
        self, query: str, run_manager: CallbackManagerForToolRun | None = None
    ) -> str:
        """Ищет записи в FAQ и возвращает найденные результаты."""
        logger.info(f"Запуск SearchFAQTool с запросом: {query}")
        results_str = "В базе знаний FAQ не найдено релевантных записей."

        try:
            with connection.get_db_session() as db:
                if not db:
                    return "Ошибка: Не удалось получить сессию базы данных."

                # Используем функцию поиска из crud.py
                # TODO: Заменить на векторный поиск позже
                search_results: List[models.FAQEntry] = crud.search_faq_entries(
                    db, query, limit=3  # Ограничим пока 3 результатами
                )

                if search_results:
                    results_str = "Найдены следующие записи в FAQ:\n\n"
                    for entry in search_results:
                        results_str += f"Q: {entry.question}\nA: {entry.answer}\n\n"

            return results_str.strip()

        except Exception as e:
            logger.error(
                f"Ошибка в SearchFAQTool при запросе '{query}': {e}", exc_info=True
            )
            return f"Произошла ошибка при поиске в FAQ: {e}"
