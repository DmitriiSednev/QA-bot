import logging
from typing import List, Optional, Dict, Any
import os  # Для переменных окружения
from datetime import datetime, timedelta

from sqlalchemy.orm import Session
from sqlalchemy import select, update as sql_update, delete as sql_delete, text, func
from sqlalchemy.exc import SQLAlchemyError
from pgvector.sqlalchemy import Vector  # Уже импортирован, но для ясности

from . import models
from .embeddings import get_embedding, get_embeddings

logger = logging.getLogger(__name__)

def add_faq_entry(db: Session, question: str, answer: str) -> Optional[models.FAQEntry]:
    """Добавляет новую запись FAQ и её векторное представление."""
    embedding = get_embedding(question)
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
    except SQLAlchemyError as e:
        logger.error(f"Ошибка SQLAlchemy при добавлении FAQ ('{question[:50]}...'): {e}", exc_info=True)
        db.rollback()
        return None
    except Exception as e:
        logger.error(f"Неожиданная ошибка при добавлении FAQ ('{question[:50]}...'): {e}", exc_info=True)
        db.rollback()
        return None


def get_faq_entry_by_id(db: Session, entry_id: int) -> Optional[models.FAQEntry]:
    """Получает запись FAQ по её ID."""
    try:
        result = db.get(models.FAQEntry, entry_id)
        return result
    except SQLAlchemyError as e:
        logger.error(
            f"Ошибка SQLAlchemy при получении FAQ ID={entry_id}: {e}", exc_info=True
        )
        return None


def search_faq_entries(
    db: Session, query: str, limit: int = 5, min_similarity: float = 0.7
) -> List[models.FAQEntry]:
    """Ищет записи FAQ, используя векторное сходство по вопросу."""
    query_embedding = get_embedding(query)
    if query_embedding is None:
        logger.error(
            f"Не удалось получить эмбеддинг для поискового запроса FAQ: '{query[:100]}...'"
        )
        return []

    try:
        max_distance = 1.0 - min_similarity

        stmt = (
            select(models.FAQEntry)
            .where(models.FAQEntry.embedding != None)
            .where(models.FAQEntry.embedding.cosine_distance(query_embedding) <= max_distance)
            .order_by(models.FAQEntry.embedding.cosine_distance(query_embedding))
            .limit(limit)
        )
        results = db.execute(stmt).scalars().all()
        logger.info(f"Найдено {len(results)} FAQ по векторному поиску для '{query[:50]}...' с порогом {min_similarity} (расстояние <= {max_distance:.4f})")
        return results
    except Exception as e:
        if "operator does not exist: vector" in str(e).lower():
            logger.error(
                f"Ошибка векторного поиска FAQ: оператор не найден. Расширение pgvector включено? Ошибка: {e}",
                exc_info=True,
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
        return None

    try:
        db_entry = db.get(models.FAQEntry, entry_id)
        if not db_entry:
            logger.warning(f"Запись FAQ с ID={entry_id} не найдена для обновления.")
            return None

        updated = False
        new_embedding: Optional[List[float]] = None

        if question is not None and db_entry.question != question:
            logger.info(f"Обновление вопроса для FAQ ID={entry_id}.")
            db_entry.question = question
            updated = True
            logger.info(f"Пересчет эмбеддинга для обновленного вопроса FAQ ID={entry_id}...")
            new_embedding = get_embedding(db_entry.question)
            if new_embedding is None:
                logger.error(f"Не удалось пересчитать эмбеддинг для FAQ ID={entry_id}. Эмбеддинг не будет обновлен.")
            else:
                db_entry.embedding = new_embedding

        if answer is not None and db_entry.answer != answer:
            logger.info(f"Обновление ответа для FAQ ID={entry_id}.")
            db_entry.answer = answer
            updated = True

        if updated:
            db.commit()
            db.refresh(db_entry)
            logger.info(f"Запись FAQ ID={entry_id} успешно обновлена.")
        else:
            logger.info(
                f"Для записи FAQ ID={entry_id} не было реальных изменений."
            )

        return db_entry
    except SQLAlchemyError as e:
        logger.error(f"Ошибка SQLAlchemy при обновлении FAQ ID={entry_id}: {e}", exc_info=True)
        db.rollback()
        return None
    except Exception as e:
        logger.error(f"Неожиданная ошибка при обновлении FAQ ID={entry_id}: {e}", exc_info=True)
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
    except SQLAlchemyError as e:
        logger.error(f"Ошибка SQLAlchemy при удалении FAQ ID={entry_id}: {e}", exc_info=True)
        db.rollback()
        return False
    except Exception as e:
        logger.error(f"Неожиданная ошибка при удалении FAQ ID={entry_id}: {e}", exc_info=True)
        db.rollback()
        return False


def get_admin_by_user_id(db: Session, user_id: int) -> Optional[models.Admin]:
    """Получает админа по его Telegram user_id."""
    try:
        stmt = select(models.Admin).where(models.Admin.user_id == user_id)
        return db.execute(stmt).scalar_one_or_none()
    except SQLAlchemyError as e:
        logger.error(f"Ошибка SQLAlchemy при получении админа по user_id={user_id}: {e}", exc_info=True)
        return None


def add_admin(db: Session, user_id: int, username: Optional[str] = None) -> Optional[models.Admin]:
    """Добавляет нового админа или активирует существующего неактивного."""
    try:
        existing_admin = get_admin_by_user_id(db, user_id)
        if existing_admin:
            if not existing_admin.is_active:
                logger.warning(f"Админ user_id={user_id} уже существует, но неактивен. Активируем и обновляем username...")
                existing_admin.is_active = True
                existing_admin.username = username
                db.commit()
                db.refresh(existing_admin)
                logger.info(f"Админ user_id={user_id} активирован.")
                return existing_admin
            else:
                if existing_admin.username != username:
                    logger.info(f"Обновление username для существующего активного админа user_id={user_id}.")
                    existing_admin.username = username
                    db.commit()
                    db.refresh(existing_admin)
                else:
                    logger.info(f"Админ user_id={user_id} уже существует и активен.")
                return existing_admin

        logger.info(f"Создание нового админа: user_id={user_id}, username={username}")
        admin = models.Admin(user_id=user_id, username=username, is_active=True)
        db.add(admin)
        db.commit()
        db.refresh(admin)
        logger.info(f"Новый админ user_id={user_id} успешно добавлен.")
        return admin
    except SQLAlchemyError as e:
        logger.error(f"Ошибка SQLAlchemy при добавлении/активации админа user_id={user_id}: {e}", exc_info=True)
        db.rollback()
        return None
    except Exception as e:
        logger.error(f"Неожиданная ошибка при добавлении/активации админа user_id={user_id}: {e}", exc_info=True)
        db.rollback()
        return None


def get_all_active_admins(db: Session) -> List[models.Admin]:
    """Получает список всех активных админов."""
    try:
        stmt = select(models.Admin).where(models.Admin.is_active == True)
        return db.execute(stmt).scalars().all()
    except SQLAlchemyError as e:
        logger.error(f"Ошибка SQLAlchemy при получении списка админов: {e}", exc_info=True)
        return []


def deactivate_admin(db: Session, user_id: int) -> bool:
    """Деактивирует админа."""
    try:
        stmt = (
            sql_update(models.Admin)
            .where(models.Admin.user_id == user_id)
            .where(models.Admin.is_active == True)
            .values(is_active=False)
        )
        result = db.execute(stmt)
        if result.rowcount > 0:
            db.commit()
            logger.info(f"Админ user_id={user_id} успешно деактивирован.")
            return True
        else:
            admin_exists = db.execute(select(models.Admin.id).where(models.Admin.user_id == user_id)).scalar()
            if admin_exists:
                logger.warning(f"Админ user_id={user_id} уже был неактивен.")
                return True
            else:
                logger.warning(f"Админ user_id={user_id} не найден для деактивации.")
                return False
    except SQLAlchemyError as e:
        logger.error(f"Ошибка SQLAlchemy при деактивации админа user_id={user_id}: {e}", exc_info=True)
        db.rollback()
        return False
    except Exception as e:
        logger.error(f"Неожиданная ошибка при деактивации админа user_id={user_id}: {e}", exc_info=True)
        db.rollback()
        return False


class EmbeddingCache:
    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, List[float]] = {}
        self._keys_in_order: List[str] = []
        self._max_size = max_size
        if max_size <= 0:
            logger.warning("Размер кэша эмбеддингов должен быть > 0. Установлен размер 1.")
            self._max_size = 1

    def get(self, text: str) -> Optional[List[float]]:
        if text in self._cache:
            self._keys_in_order.remove(text)
            self._keys_in_order.append(text)
            return self._cache[text]
        return None

    def set(self, text: str, embedding: List[float]):
        if text in self._cache:
            self._keys_in_order.remove(text)
            self._keys_in_order.append(text)
            self._cache[text] = embedding
            return

        if len(self._cache) >= self._max_size:
            oldest_key = self._keys_in_order.pop(0)
            del self._cache[oldest_key]

        self._keys_in_order.append(text)
        self._cache[text] = embedding

    def clear(self):
        self._cache.clear()
        self._keys_in_order.clear()
        logger.info("Кэш эмбеддингов очищен.")

embedding_cache = EmbeddingCache()

def get_faq_entries(db: Session, limit: int = 10, offset: int = 0) -> List[models.FAQEntry]:
    try:
        stmt = select(models.FAQEntry).order_by(models.FAQEntry.created_at.desc()).offset(offset).limit(limit)
        return db.execute(stmt).scalars().all()
    except SQLAlchemyError as e:
        logger.error(f"Ошибка SQLAlchemy при получении FAQ записей: {e}", exc_info=True)
        return []

def get_admin_by_username(db: Session, username: str) -> Optional[models.Admin]:
    if not username: return None
    try:
        stmt = select(models.Admin).where(models.Admin.username == username)
        return db.execute(stmt).scalar_one_or_none()
    except SQLAlchemyError as e:
        logger.error(f"Ошибка SQLAlchemy при получении админа по username='{username}': {e}", exc_info=True)
        return None

def cleanup_old_faq_entries(db: Session, days: int = 365) -> int:
    if days <= 0:
        logger.warning("Количество дней для очистки FAQ должно быть > 0.")
        return 0
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        stmt = sql_delete(models.FAQEntry).where(models.FAQEntry.created_at < cutoff_date)
        result = db.execute(stmt)
        deleted_count = result.rowcount
        db.commit()
        if deleted_count > 0:
            logger.info(f"Удалено {deleted_count} записей FAQ старше {days} дней.")
        else:
            logger.info(f"Не найдено записей FAQ старше {days} дней для удаления.")
        return deleted_count
    except SQLAlchemyError as e:
        logger.error(f"Ошибка SQLAlchemy при очистке старых FAQ записей: {e}", exc_info=True)
        db.rollback()
        return 0
    except Exception as e:
        logger.error(f"Неожиданная ошибка при очистке старых FAQ записей: {e}", exc_info=True)
        db.rollback()
        return 0

class ChatHistoryCache:
    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self._cache: Dict[str, Any] = {}
        self._keys_in_order: List[str] = []
        self._timestamps: Dict[str, datetime] = {}
        self._max_size = max_size
        self._ttl = timedelta(seconds=ttl_seconds)
        if max_size <= 0: self._max_size = 1
        if ttl_seconds <= 0: self._ttl = timedelta.max

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            if datetime.now() - self._timestamps[key] < self._ttl:
                self._keys_in_order.remove(key)
                self._keys_in_order.append(key)
                return self._cache[key]
            else:
                self._keys_in_order.remove(key)
                del self._cache[key]
                del self._timestamps[key]
        return None

    def set(self, key: str, data: Any):
        if key in self._cache:
            self._keys_in_order.remove(key)
            self._keys_in_order.append(key)
        elif len(self._cache) >= self._max_size:
            oldest_key = self._keys_in_order.pop(0)
            del self._cache[oldest_key]
            del self._timestamps[oldest_key]
            self._keys_in_order.append(key)
        else:
            self._keys_in_order.append(key)

        self._cache[key] = data
        self._timestamps[key] = datetime.now()

    def clear(self):
        self._cache.clear()
        self._keys_in_order.clear()
        self._timestamps.clear()
        logger.info("Кэш истории чата очищен.")

chat_history_cache = ChatHistoryCache()

def add_chat_history(
    db: Session,
    chat_id: int,
    user_id: int,
    message_text: str,
    response_text: Optional[str] = None,
    is_answered: bool = False
) -> Optional[models.ChatHistory]:
    try:
        embedding = get_embedding(message_text)
        if embedding is None:
            logger.warning(f"Не удалось получить эмбеддинг для сообщения истории чата: '{message_text[:100]}...'. Запись будет добавлена БЕЗ эмбеддинга.")

        db_entry = models.ChatHistory(
            chat_id=chat_id,
            user_id=user_id,
            message_text=message_text,
            response_text=response_text,
            embedding=embedding,
            is_answered=is_answered or (response_text is not None)
        )
        db.add(db_entry)
        db.commit()
        db.refresh(db_entry)
        return db_entry
    except SQLAlchemyError as e:
        logger.error(f"Ошибка SQLAlchemy при добавлении записи в историю чата: {e}", exc_info=True)
        db.rollback()
        return None
    except Exception as e:
        logger.error(f"Неожиданная ошибка при добавлении записи в историю чата: {e}", exc_info=True)
        db.rollback()
        return None

def search_chat_history(
    db: Session,
    query: str,
    chat_id: Optional[int] = None,
    limit: int = 5,
    min_similarity: float = 0.7,
    time_window_days: Optional[int] = None
) -> List[models.ChatHistory]:
    try:
        query_embedding = get_embedding(query)
        if query_embedding is None:
            logger.error(f"Не удалось получить эмбеддинг для запроса истории чата: '{query[:100]}...'")
            return []

        max_distance = 1.0 - min_similarity

        stmt = select(models.ChatHistory).where(models.ChatHistory.embedding != None)

        stmt = stmt.where(models.ChatHistory.embedding.cosine_distance(query_embedding) <= max_distance)

        if chat_id is not None:
            stmt = stmt.where(models.ChatHistory.chat_id == chat_id)

        if time_window_days is not None and time_window_days > 0:
            cutoff_date = datetime.now() - timedelta(days=time_window_days)
            stmt = stmt.where(models.ChatHistory.created_at >= cutoff_date)

        stmt = (
            stmt.order_by(models.ChatHistory.embedding.cosine_distance(query_embedding))
            .limit(limit)
        )

        results = db.execute(stmt).scalars().all()
        search_scope = f"в чате {chat_id}" if chat_id else "во всех чатах"
        time_scope = f" за последние {time_window_days} дней" if time_window_days else ""
        logger.info(f"Найдено {len(results)} записей в истории чата {search_scope}{time_scope} для '{query[:50]}...' (порог {min_similarity})")
        return results
    except SQLAlchemyError as e:
        logger.error(f"Ошибка SQLAlchemy при поиске в истории чата: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Неожиданная ошибка при поиске в истории чата: {e}", exc_info=True)
        return []

def update_chat_history_response(
    db: Session,
    entry_id: int,
    response_text: str,
    is_answered: bool = True,
    answer_quality: Optional[int] = None
) -> Optional[models.ChatHistory]:
    try:
        values_to_update: Dict[str, Any] = {
            "response_text": response_text,
            "is_answered": is_answered,
            "updated_at": func.now()
        }
        if answer_quality is not None:
            values_to_update["answer_quality"] = answer_quality

        stmt = (
            sql_update(models.ChatHistory)
            .where(models.ChatHistory.id == entry_id)
            .values(**values_to_update)
            .returning(models.ChatHistory)
        )
        result = db.execute(stmt).scalar_one_or_none()

        if result:
            db.commit()
            logger.info(f"Обновлен ответ для записи истории чата ID={entry_id}")
            return result
        else:
            logger.warning(f"Запись истории чата ID={entry_id} не найдена для обновления.")
            return None
    except SQLAlchemyError as e:
        logger.error(f"Ошибка SQLAlchemy при обновлении ответа в истории чата ID={entry_id}: {e}", exc_info=True)
        db.rollback()
        return None
    except Exception as e:
        logger.error(f"Неожиданная ошибка при обновлении ответа в истории чата ID={entry_id}: {e}", exc_info=True)
        db.rollback()
        return None

def add_chat_history_batch(
    db: Session,
    entries: List[Dict[str, Any]]
) -> List[Optional[models.ChatHistory]]:
    results: List[Optional[models.ChatHistory]] = []
    if not entries:
        return results

    messages = [entry.get('message_text', '') for entry in entries]
    embeddings = get_embeddings(messages) if messages else []

    if embeddings is None:
        logger.error("Ошибка получения эмбеддингов для пакетного добавления истории чата. Добавление без эмбеддингов.")
        embeddings_map = {}
    elif len(embeddings) != len(messages):
        logger.error(f"Количество эмбеддингов ({len(embeddings)}) не совпадает с количеством сообщений ({len(messages)}) для пакетного добавления.")
        embeddings_map = {msg: None for msg in messages}
    else:
        embeddings_map = dict(zip(messages, embeddings))

    db_entries_to_add = []
    for entry in entries:
        message_text = entry.get('message_text', '')
        embedding = embeddings_map.get(message_text)

        db_entry = models.ChatHistory(
            chat_id=entry.get('chat_id'),
            user_id=entry.get('user_id'),
            message_text=message_text,
            response_text=entry.get('response_text'),
            embedding=embedding,
            is_answered=entry.get('is_answered', (entry.get('response_text') is not None)),
        )
        db_entries_to_add.append(db_entry)

    if not db_entries_to_add:
        logger.warning("Нет записей для пакетного добавления в историю чата.")
        return []

    try:
        db.add_all(db_entries_to_add)
        db.commit()
        for i, entry_obj in enumerate(db_entries_to_add):
            try:
                db.refresh(entry_obj)
                results.append(entry_obj)
            except Exception as refresh_err:
                logger.error(f"Ошибка при обновлении объекта ChatHistory после batch add: {refresh_err}", exc_info=True)
                results.append(None)

        logger.info(f"Пакетно добавлено {len(results)} записей в историю чата.")
        return results

    except SQLAlchemyError as e:
        logger.error(f"Ошибка SQLAlchemy при пакетном добавлении записей в историю чата: {e}", exc_info=True)
        db.rollback()
        return [None] * len(entries)
    except Exception as e:
        logger.error(f"Неожиданная ошибка при пакетном добавлении записей в историю чата: {e}", exc_info=True)
        db.rollback()
        return [None] * len(entries)

def get_chat_history(
    db: Session,
    chat_id: int,
    limit: int = 10,
    offset: int = 0
) -> List[models.ChatHistory]:
    try:
        stmt = (
            select(models.ChatHistory)
            .where(models.ChatHistory.chat_id == chat_id)
            .order_by(models.ChatHistory.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = db.execute(stmt).scalars().all()
        return result
    except SQLAlchemyError as e:
        logger.error(f"Ошибка SQLAlchemy при получении истории чата {chat_id}: {e}", exc_info=True)
        return []

def cleanup_chat_history(
    db: Session,
    days: int = 30,
    batch_size: int = 1000
) -> int:
    if days <= 0:
        logger.warning("Количество дней для очистки истории чата должно быть > 0.")
        return 0
    if batch_size <= 0:
        logger.warning("Размер пакета для очистки истории чата должен быть > 0.")
        return 0

    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        total_deleted = 0
        logger.info(f"Начало очистки истории чата старше {cutoff_date.strftime('%Y-%m-%d')}...")

        while True:
            ids_to_delete_stmt = (
                select(models.ChatHistory.id)
                .where(models.ChatHistory.created_at < cutoff_date)
                .limit(batch_size)
            )
            ids_result = db.execute(ids_to_delete_stmt).scalars().all()

            if not ids_result:
                logger.info("Больше нет записей истории чата для удаления.")
                break

            delete_stmt = sql_delete(models.ChatHistory).where(models.ChatHistory.id.in_(ids_result))
            result = db.execute(delete_stmt)
            deleted_in_batch = result.rowcount

            try:
                db.commit()
                total_deleted += deleted_in_batch
                logger.info(f"Удалено {deleted_in_batch} записей истории чата в пакете. Всего: {total_deleted}.")
            except SQLAlchemyError as commit_err:
                logger.error(f"Ошибка SQLAlchemy при коммите удаления пакета истории чата: {commit_err}", exc_info=True)
                db.rollback()
                logger.warning("Откат транзакции удаления пакета. Остановка очистки.")
                return total_deleted

            if deleted_in_batch < batch_size:
                break

        logger.info(f"Завершена очистка истории чата старше {days} дней. Всего удалено: {total_deleted}.")
        return total_deleted
    except SQLAlchemyError as e:
        logger.error(f"Ошибка SQLAlchemy при очистке истории чата: {e}", exc_info=True)
        db.rollback()
        return -1
    except Exception as e:
        logger.error(f"Неожиданная ошибка при очистке истории чата: {e}", exc_info=True)
        db.rollback()
        return -1

def analyze_chat_history(
    db: Session,
    chat_id: int,
    time_window_days: Optional[int] = None
) -> Dict[str, Any]:
    try:
        subq = select(models.ChatHistory).where(models.ChatHistory.chat_id == chat_id)
        if time_window_days is not None and time_window_days > 0:
            cutoff_date = datetime.now() - timedelta(days=time_window_days)
            subq = subq.where(models.ChatHistory.created_at >= cutoff_date)
        subq = subq.subquery()

        total_messages_stmt = select(func.count()).select_from(subq)
        answered_stmt = select(func.count()).select_from(subq).where(subq.c.is_answered == True)
        avg_response_time_stmt = select(
            func.coalesce(
                func.avg(
                    func.extract('epoch', subq.c.updated_at - subq.c.created_at)
                ),
                0.0
            )
        ).select_from(subq).where(subq.c.is_answered == True)

        total_messages = db.execute(total_messages_stmt).scalar() or 0
        answered_messages = db.execute(answered_stmt).scalar() or 0
        avg_response_time = db.execute(avg_response_time_stmt).scalar()

        return {
            'total_messages': total_messages,
            'answered_messages': answered_messages,
            'unanswered_messages': total_messages - answered_messages,
            'response_rate': (answered_messages / total_messages * 100.0) if total_messages > 0 else 0.0,
            'avg_response_time_seconds': float(avg_response_time),
        }
    except SQLAlchemyError as e:
        logger.error(f"Ошибка SQLAlchemy при анализе истории чата {chat_id}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Неожиданная ошибка при анализе истории чата {chat_id}: {e}", exc_info=True)

    return {
        'error': "Произошла ошибка при анализе истории чата",
        'total_messages': 0,
        'answered_messages': 0,
        'unanswered_messages': 0,
        'response_rate': 0.0,
        'avg_response_time_seconds': 0.0,
    }
