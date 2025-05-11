import logging
from typing import List, Optional
import os  # Для переменных окружения

from sqlalchemy.orm import Session
from sqlalchemy import select, update as sql_update, delete as sql_delete, text
from sqlalchemy.exc import SQLAlchemyError
from langchain_openai import OpenAIEmbeddings  # Импортируем эмбеддер
from pgvector.sqlalchemy import Vector  # Уже импортирован, но для ясности

from . import models

logger = logging.getLogger(__name__)

# --- Инициализация Эмбеддера ---
# Использует api_key и base_url из .env
try:
    # Убедимся, что нужные переменные есть
    proxy_api_key = os.getenv("api_key")  # Используем новое имя
    proxy_base_url = os.getenv("base_url")  # Используем новое имя

    if not proxy_api_key or not proxy_base_url:
        raise ValueError(
            "Ключ API (api_key) и базовый URL (base_url) для OpenAI-совместимого API "
            "должны быть указаны в .env для эмбеддингов."
        )
    embeddings_model = OpenAIEmbeddings(
        # Можно указать конкретную модель, если нужно, иначе по умолчанию
        # model="text-embedding-ada-002"
        openai_api_key=proxy_api_key,  # Передаем proxy_api_key
        openai_api_base=proxy_base_url,  # Передаем proxy_base_url
    )
    logger.info(
        f"Модель эмбеддингов OpenAI инициализирована с базовым URL: {proxy_base_url}."
    )
except Exception as e:
    logger.error(f"Ошибка инициализации модели эмбеддингов: {e}", exc_info=True)
    embeddings_model = None  # Устанавливаем в None, чтобы CRUD функции могли проверить


def _get_embedding(text: str) -> Optional[List[float]]:
    """Вспомогательная функция для получения эмбеддинга текста."""
    if not embeddings_model:
        logger.error(
            "Модель эмбеддингов не инициализирована, не могу получить эмбеддинг."
        )
        return None
    try:
        return embeddings_model.embed_query(text)
    except Exception as e:
        logger.error(
            f"Ошибка при получении эмбеддинга для текста: {text[:100]}...: {e}",
            exc_info=True,
        )
        return None


def add_faq_entry(db: Session, question: str, answer: str) -> Optional[models.FAQEntry]:
    """Добавляет новую запись FAQ и её векторное представление."""
    embedding = _get_embedding(question)  # Получаем эмбеддинг для вопроса
    if embedding is None:
        logger.error(f"Не удалось получить эмбеддинг для вопроса: {question[:100]}...")
        # Можно решить, добавлять ли запись без эмбеддинга или нет.
        # Пока что не будем добавлять, если эмбеддинг не сгенерирован.
        return None

    try:
        db_entry = models.FAQEntry(
            question=question, answer=answer, embedding=embedding  # Сохраняем эмбеддинг
        )
        db.add(db_entry)
        db.commit()
        db.refresh(db_entry)
        logger.info(f"Добавлена запись FAQ ID={db_entry.id} с эмбеддингом.")
        return db_entry
    except Exception as e:
        logger.error(f"Ошибка при добавлении FAQ ({question}): {e}", exc_info=True)
        db.rollback()
        return None


def get_faq_entry_by_id(db: Session, entry_id: int) -> Optional[models.FAQEntry]:
    """Получает запись FAQ по её ID."""
    try:
        statement = select(models.FAQEntry).where(models.FAQEntry.id == entry_id)
        result = db.execute(statement).scalar_one_or_none()
        return result
    except SQLAlchemyError as e:
        logger.error(
            f"Ошибка SQLAlchemy при получении FAQ ID={entry_id}: {e}", exc_info=True
        )
        return None


def search_faq_entries(
    db: Session, query: str, limit: int = 5
) -> List[models.FAQEntry]:
    """Ищет записи FAQ, используя векторное сходство по вопросу."""
    query_embedding = _get_embedding(query)
    if query_embedding is None:
        logger.error(
            f"Не удалось получить эмбеддинг для поискового запроса: {query[:100]}..."
        )
        return []

    try:
        # Используем векторный поиск по косинусному расстоянию (<=>)
        # Чем меньше расстояние, тем ближе векторы (более похожи)
        stmt = (
            select(models.FAQEntry)
            .order_by(models.FAQEntry.embedding.cosine_distance(query_embedding))
            .limit(limit)
        )
        results = db.execute(stmt).scalars().all()
        logger.info(f"Найдено {len(results)} FAQ по векторному поиску для '{query}'")
        return results
    except Exception as e:
        # Конкретизируем ошибку, если это ошибка отсутствия оператора (расширение не включено?)
        if "operator does not exist: vector <=> vector" in str(e):
            logger.error(
                f"Ошибка векторного поиска: оператор <=> не найден. Расширение pgvector включено и настроено правильно? Ошибка: {e}",
                exc_info=True,
            )
        else:
            logger.error(
                f"Ошибка при векторном поиске FAQ по запросу '{query}': {e}",
                exc_info=True,
            )
        return []


def update_faq_entry(
    db: Session, entry_id: int, question: str | None = None, answer: str | None = None
) -> Optional[models.FAQEntry]:
    """Обновляет запись FAQ и её векторное представление (если вопрос изменен)."""
    try:
        db_entry = db.get(models.FAQEntry, entry_id)
        if not db_entry:
            logger.warning(f"Запись FAQ с ID={entry_id} не найдена для обновления.")
            return None

        updated = False
        embedding_needs_update = False
        if question is not None and db_entry.question != question:
            db_entry.question = question
            updated = True
            embedding_needs_update = True  # Вопрос изменился, нужен новый эмбеддинг
        if answer is not None and db_entry.answer != answer:
            db_entry.answer = answer
            updated = True
            # Решаем, нужно ли пересчитывать эмбеддинг при изменении ответа.
            # Если эмбеддинг только по вопросу, то нет.
            # Если по вопросу+ответу, то embedding_needs_update = True

        if updated:
            if embedding_needs_update:
                logger.info(
                    f"Пересчет эмбеддинга для обновленного вопроса FAQ ID={entry_id}..."
                )
                new_embedding = _get_embedding(db_entry.question)
                if new_embedding:
                    db_entry.embedding = new_embedding
                else:
                    logger.error(
                        f"Не удалось пересчитать эмбеддинг для FAQ ID={entry_id}. Эмбеддинг не будет обновлен."
                    )
                    # Можно решить, откатывать ли транзакцию или сохранить без эмбеддинга

            db.commit()
            db.refresh(db_entry)
            logger.info(f"Запись FAQ ID={entry_id} обновлена.")
        else:
            logger.info(
                f"Для записи FAQ ID={entry_id} не было указано полей для обновления."
            )

        return db_entry
    except Exception as e:
        logger.error(f"Ошибка при обновлении FAQ ID={entry_id}: {e}", exc_info=True)
        db.rollback()
        return None


def delete_faq_entry(db: Session, entry_id: int) -> bool:
    """Удаляет запись FAQ по ID."""
    try:
        db_entry = db.get(models.FAQEntry, entry_id)
        if not db_entry:
            logger.warning(f"Запись FAQ с ID={entry_id} не найдена для удаления.")
            return False

        # TODO: Удалить соответствующий вектор из векторного хранилища, ЕСЛИ оно отдельное.
        # Если вектор в той же таблице, он удалится вместе с записью.
        db.delete(db_entry)
        db.commit()
        logger.info(f"Запись FAQ ID={entry_id} удалена.")
        return True
    except Exception as e:
        logger.error(f"Ошибка при удалении FAQ ID={entry_id}: {e}", exc_info=True)
        db.rollback()
        return False
