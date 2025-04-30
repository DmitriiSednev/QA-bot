import os
import logging
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
from langchain_openai import OpenAIEmbeddings
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String, Text, DateTime, func
from sqlalchemy.orm import Session

from .models import Base
from .connection import get_db_session

logger = logging.getLogger(__name__)

class YandexDocsEntry(Base):
    """Модель для хранения документации Yandex Cloud."""
    __tablename__ = "yandex_docs_entries"

    id = Column(Integer, primary_key=True, index=True)
    path = Column(String(500), nullable=False, unique=True)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=True)
    last_updated = Column(DateTime, default=func.now(), onupdate=func.now())
    created_at = Column(DateTime, default=func.now())

class YandexDocsManager:
    """Менеджер для работы с документацией Yandex Cloud."""

    def __init__(self):
        self.base_url = "https://github.com/yandex-cloud/docs/tree/master/ru/foundation-models"
        self.embeddings_model = OpenAIEmbeddings(
            openai_api_key=os.getenv("API_KEY"),
            openai_api_base=os.getenv("API_BASE"),
            model="text-embedding-ada-002"
        )

    def _clean_content(self, content: str) -> str:
        """Очищает контент от специальных символов и форматирования."""
        # Заменяем специальные разделители на пробелы
        content = re.sub(r'\|\|', ' ', content)
        # Заменяем множественные пробелы на один
        content = re.sub(r'\s+', ' ', content)
        return content.strip()

    def _parse_table(self, table_content: str) -> List[Dict[str, str]]:
        """Парсит содержимое таблицы."""
        rows = []
        for line in table_content.split('\n'):
            if '|' in line:
                parts = [part.strip() for part in line.split('|') if part.strip()]
                if len(parts) >= 2:
                    rows.append({
                        "field": parts[0],
                        "description": ' '.join(parts[1:])
                    })
        return rows

    async def update_docs(self, session: Session):
        """Обновляет локальную копию документации."""
        try:
            # TODO: Реализовать загрузку документации с GitHub
            # Для каждого файла:
            # 1. Получить содержимое
            # 2. Распарсить
            # 3. Создать эмбеддинги
            # 4. Сохранить в БД
            pass
        except Exception as e:
            logger.error(f"Ошибка при обновлении документации: {e}", exc_info=True)

    def search_docs(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Ищет информацию в документации."""
        try:
            # Получаем эмбеддинг запроса
            query_embedding = self.embeddings_model.embed_query(query)

            with get_db_session() as db:
                # Используем векторный поиск
                stmt = (
                    db.query(YandexDocsEntry)
                    .order_by(YandexDocsEntry.embedding.cosine_distance(query_embedding))
                    .limit(limit)
                )
                results = stmt.all()

                return [{
                    "title": entry.title,
                    "content": self._clean_content(entry.content),
                    "path": entry.path,
                    "last_updated": entry.last_updated
                } for entry in results]

        except Exception as e:
            logger.error(f"Ошибка при поиске в документации: {e}", exc_info=True)
            return []

    def format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Форматирует результаты поиска в читаемый вид."""
        if not results:
            return "В документации не найдено релевантной информации."

        formatted = "Найдена следующая информация в документации Yandex Cloud:\n\n"
        for result in results:
            formatted += f"### {result['title']}\n"
            formatted += f"{result['content']}\n"
            formatted += f"Источник: {result['path']}\n\n"

        return formatted 