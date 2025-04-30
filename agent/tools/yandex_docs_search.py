import logging
from typing import Type, Optional, Dict, Any
import re
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

from database.yandex_docs import YandexDocsManager

logger = logging.getLogger(__name__)

class YandexDocsSearchInput(BaseModel):
    """Схема входных данных для YandexDocsSearchTool."""
    query: str = Field(description="Поисковый запрос для поиска в документации Yandex Cloud")

class YandexDocsSearchTool(BaseTool):
    """Инструмент для поиска информации в документации Yandex Cloud."""

    name: str = "yandex_docs_search"
    description: str = (
        "Используй этот инструмент для поиска информации в документации Yandex Cloud. "
        "Он может обрабатывать специальные форматы документации, включая таблицы, "
        "код и технические символы. Особенно полезен для поиска информации об API, "
        "параметрах и технических деталях сервисов Yandex Cloud."
    )
    args_schema: Type[BaseModel] = YandexDocsSearchInput
    docs_manager: YandexDocsManager = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.docs_manager = YandexDocsManager()

    def _clean_text(self, text: str) -> str:
        """Очищает текст от специальных символов и форматирования."""
        # Заменяем специальные разделители на пробелы
        text = re.sub(r'\|\|', ' ', text)
        # Заменяем множественные пробелы на один
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _parse_table_row(self, row: str) -> Dict[str, str]:
        """Парсит строку таблицы в словарь."""
        parts = [part.strip() for part in row.split('|') if part.strip()]
        if len(parts) >= 2:
            return {
                "field": parts[0],
                "description": ' '.join(parts[1:])
            }
        return {}

    def _format_response(self, content: str) -> str:
        """Форматирует найденную информацию в читаемый вид."""
        # Если это похоже на строку таблицы
        if '|' in content:
            parsed = self._parse_table_row(content)
            if parsed:
                return f"Поле: {parsed['field']}\nОписание: {parsed['description']}"
        
        # Для обычного текста
        return self._clean_text(content)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Выполняет поиск в документации."""
        logger.info(f"Поиск в документации Yandex Cloud: {query}")

        try:
            # Ищем в документации
            results = self.docs_manager.search_docs(query)
            
            # Форматируем результаты
            return self.docs_manager.format_search_results(results)

        except Exception as e:
            logger.error(f"Ошибка при поиске в документации: {e}", exc_info=True)
            return (
                f"Произошла ошибка при поиске в документации: {e}\n"
                "Пожалуйста, попробуйте позже или обратитесь к официальной "
                "документации: https://cloud.yandex.ru/docs"
            ) 