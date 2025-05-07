import logging
from typing import List, Optional
from datetime import datetime, timedelta

from sqlalchemy.orm import Session
from sqlalchemy import select, delete as sql_delete, text
from pgvector.sqlalchemy import Vector

from . import models
# Используем относительный импорт для embeddings внутри пакета database
from .embeddings import get_embeddings

logger = logging.getLogger(__name__)

# Функции, перенесенные из crud.py, относящиеся к FAQEntry

def add_faq_entry(db: Session, question: str, answer: str) -> Optional[models.FAQEntry]:
    """Добавляет новую запись FAQ и её векторное представление."""
    embeddings_list = get_embeddings([question]) # get_embeddings ожидает список
    embedding = embeddings_list[0] if embeddings_list else None

    if embedding is None:
        logger.warning(f"Не удалось получить эмбеддинг для вопроса FAQ: '{question[:100]}...'. Запись будет добавлена БЕЗ эмбеддинга.")
        db_entry = models.FAQEntry(question=question, answer=answer, embedding=None)
    else:
        db_entry = models.FAQEntry(
            question=question, answer=answer, embedding=embedding
        )

    try:
        db.add(db_entry)
        db.commit()
        db.refresh(db_entry)
        if embedding:
            logger.info(f"Добавлена запись FAQ ID={db_entry.id} с эмбеддингом.")
        else:
            logger.warning(f"Добавлена запись FAQ ID={db_entry.id} БЕЗ эмбеддинга.")
        return db_entry
    except Exception as e: # Ловим общую ошибку на всякий случай
        logger.error(f"Ошибка при добавлении FAQ ('{question[:50]}...'): {e}", exc_info=True)
        db.rollback()
        return None


def get_faq_entry_by_id(db: Session, entry_id: int) -> Optional[models.FAQEntry]:
    """Получает запись FAQ по её ID."""
    try:
        # Используем db.get для поиска по первичному ключу
        result = db.get(models.FAQEntry, entry_id)
        return result
    except Exception as e:
        logger.error(
            f"Ошибка при получении FAQ ID={entry_id}: {e}", exc_info=True
        )
        return None


def search_faq_entries(
    db: Session, query: str, limit: int = 5, min_similarity: float = 0.7
) -> List[models.FAQEntry]:
    """Ищет записи FAQ, используя векторное сходство по вопросу."""
    query_embeddings_list = get_embeddings([query]) # get_embeddings ожидает список
    query_embedding = query_embeddings_list[0] if query_embeddings_list else None

    if query_embedding is None:
        logger.error(
            f"Не удалось получить эмбеддинг для поискового запроса FAQ: '{query[:100]}...'"
        )
        return []

    try:
        # Расстояние косинуса: 0 = идентичны, 2 = противоположны.
        # Сходство = 1 - расстояние.
        # Мы ищем расстояние <= (1 - min_similarity)
        # Например, min_similarity = 0.7 -> ищем расстояние <= 0.3
        max_distance = 1.0 - min_similarity

        stmt = (
            select(models.FAQEntry)
            .where(models.FAQEntry.embedding != None) # Ищем только записи с эмбеддингами
            # Используем оператор cosine_distance (<=>)
            .where(models.FAQEntry.embedding.cosine_distance(query_embedding) <= max_distance)
            .order_by(models.FAQEntry.embedding.cosine_distance(query_embedding)) # Сортируем по возрастанию расстояния (убыванию сходства)
            .limit(limit)
        )
        results = db.execute(stmt).scalars().all()
        logger.info(f"Найдено {len(results)} FAQ по векторному поиску для '{query[:50]}...' с порогом {min_similarity} (расстояние <= {max_distance:.4f})")
        return results
    except Exception as e:
        if "operator does not exist: vector <=> vector" in str(e).lower():
             logger.error(
                 f"Ошибка векторного поиска FAQ: оператор <=> не найден. Расширение pgvector включено? Ошибка: {e}",
                 exc_info=False, # Не логируем полный traceback для этой ожидаемой ошибки
             )
        elif "function cosine_distance" in str(e).lower():
             logger.error(
                 f"Ошибка векторного поиска FAQ: функция cosine_distance не найдена или используется неверно. Используйте оператор <=> для pgvector >= 0.5.0. Ошибка: {e}",
                 exc_info=False,
            )
        else:
            logger.error(
                f"Ошибка при векторном поиске FAQ по запросу '{query[:50]}...': {e}",
                exc_info=True,
            )
        return []


def update_faq_entry(
    db: Session, entry_id: int, question: Optional[str] = None, answer: Optional[str] = None
) -> Optional[models.FAQEntry]:
    """Обновляет запись FAQ и её векторное представление (если вопрос изменен)."""
    if question is None and answer is None:
        logger.warning(f"Попытка обновить FAQ ID={entry_id} без указания новых данных.")
        return None # Возвращаем None, если нет данных для обновления

    try:
        db_entry = db.get(models.FAQEntry, entry_id)
        if not db_entry:
            logger.warning(f"Запись FAQ с ID={entry_id} не найдена для обновления.")
            return None

        updated = False
        new_embedding_vector: Optional[List[float]] = None # Правильный тип для эмбеддинга

        if question is not None and db_entry.question != question:
            logger.info(f"Обновление вопроса для FAQ ID={entry_id}.")
            db_entry.question = question
            updated = True
            logger.info(f"Пересчет эмбеддинга для обновленного вопроса FAQ ID={entry_id}...")
            # Получаем эмбеддинг как список списков, берем первый элемент
            new_embeddings_list = get_embeddings([db_entry.question])
            new_embedding_vector = new_embeddings_list[0] if new_embeddings_list else None

            if new_embedding_vector is None:
                logger.error(f"Не удалось пересчитать эмбеддинг для FAQ ID={entry_id}. Эмбеддинг не будет обновлен.")
                # Решаем, что делать: оставить старый или обнулить? Пока оставляем старый.
            else:
                db_entry.embedding = new_embedding_vector # Присваиваем вектор

        if answer is not None and db_entry.answer != answer:
            logger.info(f"Обновление ответа для FAQ ID={entry_id}.")
            db_entry.answer = answer
            updated = True

        if updated:
            # db.add(db_entry) # Не нужно add при обновлении существующего объекта из сессии
            db.commit()
            db.refresh(db_entry)
            logger.info(f"Запись FAQ ID={entry_id} успешно обновлена.")
        else:
            logger.info(
                f"Для записи FAQ ID={entry_id} не было реальных изменений."
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
        db.delete(db_entry)
        db.commit()
        logger.info(f"Запись FAQ ID={entry_id} удалена.")
        return True
    except Exception as e:
        logger.error(f"Ошибка при удалении FAQ ID={entry_id}: {e}", exc_info=True)
        db.rollback()
        return False

def get_faq_entries(db: Session, limit: int = 10, offset: int = 0) -> List[models.FAQEntry]:
    """Получает список записей FAQ с пагинацией."""
    try:
        stmt = select(models.FAQEntry).order_by(models.FAQEntry.created_at.desc()).offset(offset).limit(limit)
        return db.execute(stmt).scalars().all()
    except Exception as e:
        logger.error(f"Ошибка SQLAlchemy при получении FAQ записей: {e}", exc_info=True)
        return []

def cleanup_old_faq_entries(db: Session, days: int = 365) -> int:
    """Удаляет старые записи FAQ."""
    if days <= 0:
        logger.warning("Количество дней для очистки FAQ должно быть > 0.")
        return 0
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        # Используем text() для динамического имени таблицы, если нужно, но лучше через ORM
        # stmt = text("DELETE FROM faq_entries WHERE created_at < :cutoff")
        # result = db.execute(stmt, {"cutoff": cutoff_date})

        # ORM-путь
        stmt = sql_delete(models.FAQEntry).where(models.FAQEntry.created_at < cutoff_date)
        result = db.execute(stmt)
        deleted_count = result.rowcount
        db.commit()
        if deleted_count > 0:
            logger.info(f"Удалено {deleted_count} записей FAQ старше {days} дней.")
        else:
            logger.info(f"Не найдено записей FAQ старше {days} дней для удаления.")
        return deleted_count
    except Exception as e:
        logger.error(f"Ошибка при очистке старых FAQ записей: {e}", exc_info=True)
        db.rollback()
        return 0 