import logging
from typing import List, Optional, Dict, Any
import os  # Для переменных окружения
from datetime import datetime, timedelta
import numpy as np
from functools import lru_cache

from sqlalchemy.orm import Session
from sqlalchemy import select, update as sql_update, delete as sql_delete, text
from sqlalchemy.exc import SQLAlchemyError
from langchain_openai import OpenAIEmbeddings
from pgvector.sqlalchemy import Vector  # Уже импортирован, но для ясности

from . import models
from .embeddings import get_embeddings

logger = logging.getLogger(__name__)

# --- Инициализация Эмбеддера ---
# Используем API_KEY и API_BASE из .env, как и для основного LLM
try:
    # Убедимся, что нужные переменные есть
    # Используем те же имена переменных, что и для основного LLM прокси
    api_key_for_embeddings = os.getenv("API_KEY")
    api_base_for_embeddings = os.getenv("API_BASE")
    if not api_key_for_embeddings or not api_base_for_embeddings:
        raise ValueError(
            "API_KEY или API_BASE не найдены в .env для эмбеддингов (используются те же, что и для LLM)"
        )
    embeddings_model = OpenAIEmbeddings(
        openai_api_key=api_key_for_embeddings,
        openai_api_base=api_base_for_embeddings,
        model="text-embedding-ada-002",  # Можно оставить эту модель для эмбеддингов
        request_timeout=30,
        headers={
            "HTTP-Referer": "https://github.com/DmitriiSednev/QA-bot",
            "X-Title": "QA Telegram Bot"
        }
    )
    logger.info("Модель эмбеддингов инициализирована (через прокси).")
except Exception as e:
    logger.error(f"Ошибка инициализации модели эмбеддингов (через прокси): {e}", exc_info=True)
    embeddings_model = None


def _get_embedding(text: str) -> Optional[List[float]]:
    """Получает эмбеддинг для текста."""
    return get_embeddings(text)


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


def cleanup_old_faq_entries(db: Session, days: int = 365) -> int:
    """Удаляет записи FAQ старше указанного количества дней."""
    try:
        # Используем text() для прямого SQL запроса с интервалом
        stmt = sql_delete(models.FAQEntry).where(
            models.FAQEntry.created_at < text(f"NOW() - INTERVAL '{days} days'")
        )
        result = db.execute(stmt)
        db.commit()
        deleted_count = result.rowcount
        logger.info(f"Удалено {deleted_count} записей FAQ старше {days} дней.")
        return deleted_count
    except Exception as e:
        logger.error(f"Ошибка при очистке старых FAQ записей: {e}", exc_info=True)
        db.rollback()
        return 0


def get_admin_by_user_id(db: Session, user_id: int) -> Optional[models.Admin]:
    """Получает админа по его Telegram user_id."""
    try:
        return db.query(models.Admin).filter(models.Admin.user_id == user_id).first()
    except Exception as e:
        logger.error(f"Ошибка при получении админа по user_id={user_id}: {e}", exc_info=True)
        return None


def add_admin(db: Session, user_id: int, username: Optional[str] = None) -> Optional[models.Admin]:
    """Добавляет нового админа."""
    try:
        admin = models.Admin(user_id=user_id, username=username)
        db.add(admin)
        db.commit()
        db.refresh(admin)
        return admin
    except Exception as e:
        logger.error(f"Ошибка при добавлении админа user_id={user_id}: {e}", exc_info=True)
        db.rollback()
        return None


def get_all_active_admins(db: Session) -> List[models.Admin]:
    """Получает список всех активных админов."""
    try:
        return db.query(models.Admin).filter(models.Admin.is_active == True).all()
    except Exception as e:
        logger.error(f"Ошибка при получении списка админов: {e}", exc_info=True)
        return []


def deactivate_admin(db: Session, user_id: int) -> bool:
    """Деактивирует админа."""
    try:
        admin = get_admin_by_user_id(db, user_id)
        if admin:
            admin.is_active = False
            db.commit()
            return True
        return False
    except Exception as e:
        logger.error(f"Ошибка при деактивации админа user_id={user_id}: {e}", exc_info=True)
        db.rollback()
        return False


class EmbeddingCache:
    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, np.ndarray] = {}
        self._max_size = max_size

    def get(self, text: str) -> Optional[np.ndarray]:
        return self._cache.get(text)

    def set(self, text: str, embedding: np.ndarray):
        if len(self._cache) >= self._max_size:
            # Удаляем самый старый элемент
            self._cache.pop(next(iter(self._cache)))
        self._cache[text] = embedding

    def clear(self):
        self._cache.clear()

embedding_cache = EmbeddingCache()

def get_faq_entries(db: Session, limit: int = 10) -> List[models.FAQEntry]:
    """Получение FAQ записей с кэшированием."""
    return db.query(models.FAQEntry).order_by(models.FAQEntry.created_at.desc()).limit(limit).all()

@lru_cache(maxsize=100)
def get_admin_by_username(username: str) -> Optional[models.Admin]:
    """Получение админа с кэшированием."""
    with connection.get_db_session() as db:
        return db.query(models.Admin).filter(models.Admin.username == username).first()

def create_faq_entry(db: Session, question: str, answer: str, category: str) -> models.FAQEntry:
    """Создание FAQ записи с оптимизированными эмбеддингами."""
    # Проверяем кэш
    cached_embedding = embedding_cache.get(question)
    if cached_embedding is not None:
        question_embedding = cached_embedding
    else:
        # Получаем эмбеддинги батчами
        embeddings = get_embeddings([question])
        question_embedding = embeddings[0]
        embedding_cache.set(question, question_embedding)

    entry = models.FAQEntry(
        question=question,
        answer=answer,
        category=category,
        question_embedding=question_embedding.tolist()
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry

def search_similar_questions(db: Session, question: str, threshold: float = 0.7, limit: int = 5) -> List[models.FAQEntry]:
    """Поиск похожих вопросов с оптимизированными эмбеддингами."""
    # Проверяем кэш
    cached_embedding = embedding_cache.get(question)
    if cached_embedding is not None:
        question_embedding = cached_embedding
    else:
        embeddings = get_embeddings([question])
        question_embedding = embeddings[0]
        embedding_cache.set(question, question_embedding)

    # Используем векторный поиск
    similar_entries = db.query(models.FAQEntry).filter(
        models.FAQEntry.question_embedding.cosine_similarity(question_embedding) > threshold
    ).order_by(
        models.FAQEntry.question_embedding.cosine_similarity(question_embedding).desc()
    ).limit(limit).all()

    return similar_entries

def cleanup_old_faq_entries(db: Session, days: int = 30) -> int:
    """Очистка старых FAQ записей с оптимизацией."""
    cutoff_date = datetime.now() - timedelta(days=days)
    deleted = db.query(models.FAQEntry).filter(models.FAQEntry.created_at < cutoff_date).delete()
    db.commit()
    
    # Очищаем кэш после удаления
    embedding_cache.clear()
    
    return deleted

class ChatHistoryCache:
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self._cache: Dict[str, Dict] = {}
        self._max_size = max_size
        self._ttl = ttl

    def get(self, key: str) -> Optional[Dict]:
        if key in self._cache:
            entry = self._cache[key]
            if datetime.now() - entry['timestamp'] < timedelta(seconds=self._ttl):
                return entry['data']
            del self._cache[key]
        return None

    def set(self, key: str, data: Dict):
        if len(self._cache) >= self._max_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]['timestamp'])
            del self._cache[oldest_key]
        self._cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        }

    def clear(self):
        self._cache.clear()

chat_history_cache = ChatHistoryCache()

def add_chat_history(
    db: Session,
    chat_id: int,
    user_id: int,
    message_text: str,
    response_text: str | None = None
) -> Optional[models.ChatHistory]:
    """Добавляет новую запись в историю чата."""
    try:
        embedding = _get_embedding(message_text)
        db_entry = models.ChatHistory(
            chat_id=chat_id,
            user_id=user_id,
            message_text=message_text,
            response_text=response_text,
            embedding=embedding
        )
        db.add(db_entry)
        db.commit()
        db.refresh(db_entry)
        return db_entry
    except Exception as e:
        logger.error(f"Ошибка при добавлении записи в историю чата: {e}", exc_info=True)
        db.rollback()
        return None

def search_chat_history(
    db: Session,
    query: str,
    limit: int = 5,
    min_similarity: float = 0.7
) -> List[models.ChatHistory]:
    """Ищет похожие вопросы в истории чата."""
    try:
        query_embedding = _get_embedding(query)
        if query_embedding is None:
            return []

        stmt = (
            select(models.ChatHistory)
            .where(models.ChatHistory.embedding.cosine_distance(query_embedding) <= 1 - min_similarity)
            .order_by(models.ChatHistory.embedding.cosine_distance(query_embedding))
            .limit(limit)
        )
        results = db.execute(stmt).scalars().all()
        return results
    except Exception as e:
        logger.error(f"Ошибка при поиске в истории чата: {e}", exc_info=True)
        return []

def update_chat_history_response(
    db: Session,
    entry_id: int,
    response_text: str,
    is_answered: bool = True
) -> Optional[models.ChatHistory]:
    """Обновляет ответ в истории чата."""
    try:
        entry = db.get(models.ChatHistory, entry_id)
        if entry:
            entry.response_text = response_text
            entry.is_answered = is_answered
            db.commit()
            db.refresh(entry)
            return entry
        return None
    except Exception as e:
        logger.error(f"Ошибка при обновлении ответа в истории чата: {e}", exc_info=True)
        db.rollback()
        return None

def add_chat_history_batch(
    db: Session,
    entries: List[Dict[str, Any]]
) -> List[Optional[models.ChatHistory]]:
    """Пакетное добавление записей в историю чата."""
    try:
        # Получаем эмбеддинги для всех сообщений сразу
        messages = [entry['message_text'] for entry in entries]
        embeddings = get_embeddings(messages)

        db_entries = []
        for entry, embedding in zip(entries, embeddings):
            db_entry = models.ChatHistory(
                chat_id=entry['chat_id'],
                user_id=entry['user_id'],
                message_text=entry['message_text'],
                response_text=entry.get('response_text'),
                embedding=embedding,
                is_answered=entry.get('is_answered', False)
            )
            db_entries.append(db_entry)

        db.add_all(db_entries)
        db.commit()
        for entry in db_entries:
            db.refresh(entry)

        return db_entries
    except Exception as e:
        logger.error(f"Ошибка при пакетном добавлении записей в историю чата: {e}", exc_info=True)
        db.rollback()
        return []

def get_chat_history(
    db: Session,
    chat_id: int,
    limit: int = 10,
    offset: int = 0
) -> List[models.ChatHistory]:
    """Получение истории чата с кэшированием."""
    cache_key = f"chat_{chat_id}_{limit}_{offset}"
    cached_result = chat_history_cache.get(cache_key)
    if cached_result is not None:
        return cached_result

    try:
        result = (
            db.query(models.ChatHistory)
            .filter(models.ChatHistory.chat_id == chat_id)
            .order_by(models.ChatHistory.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        chat_history_cache.set(cache_key, result)
        return result
    except Exception as e:
        logger.error(f"Ошибка при получении истории чата {chat_id}: {e}", exc_info=True)
        return []

def search_chat_history_optimized(
    db: Session,
    query: str,
    chat_id: Optional[int] = None,
    limit: int = 5,
    min_similarity: float = 0.7,
    time_window: Optional[int] = None  # в днях
) -> List[models.ChatHistory]:
    """Оптимизированный поиск по истории чата."""
    try:
        # Проверяем кэш для эмбеддинга запроса
        query_embedding = embedding_cache.get(query)
        if query_embedding is None:
            query_embedding = _get_embedding(query)
            if query_embedding is not None:
                embedding_cache.set(query, query_embedding)

        if query_embedding is None:
            return []

        # Строим базовый запрос
        stmt = select(models.ChatHistory).where(
            models.ChatHistory.embedding.cosine_distance(query_embedding) <= 1 - min_similarity
        )

        # Добавляем фильтры
        if chat_id is not None:
            stmt = stmt.where(models.ChatHistory.chat_id == chat_id)

        if time_window is not None:
            cutoff_date = datetime.now() - timedelta(days=time_window)
            stmt = stmt.where(models.ChatHistory.created_at >= cutoff_date)

        # Сортировка и лимит
        stmt = (
            stmt.order_by(models.ChatHistory.embedding.cosine_distance(query_embedding))
            .limit(limit)
        )

        results = db.execute(stmt).scalars().all()
        return results
    except Exception as e:
        logger.error(f"Ошибка при оптимизированном поиске в истории чата: {e}", exc_info=True)
        return []

def cleanup_chat_history(
    db: Session,
    days: int = 30,
    batch_size: int = 1000
) -> int:
    """Пакетная очистка старой истории чата."""
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        total_deleted = 0
        
        while True:
            # Получаем batch ID для удаления
            ids_to_delete = (
                db.query(models.ChatHistory.id)
                .filter(models.ChatHistory.created_at < cutoff_date)
                .limit(batch_size)
                .all()
            )
            
            if not ids_to_delete:
                break
                
            # Преобразуем список кортежей в список ID
            ids = [id_[0] for id_ in ids_to_delete]
            
            # Удаляем записи пакетом
            deleted = db.query(models.ChatHistory).filter(
                models.ChatHistory.id.in_(ids)
            ).delete(synchronize_session=False)
            
            db.commit()
            total_deleted += deleted
            
            # Очищаем кэш после каждого пакета
            chat_history_cache.clear()
            
        return total_deleted
    except Exception as e:
        logger.error(f"Ошибка при очистке истории чата: {e}", exc_info=True)
        db.rollback()
        return 0

def analyze_chat_history(
    db: Session,
    chat_id: int,
    time_window: Optional[int] = None  # в днях
) -> Dict[str, Any]:
    """Анализ истории чата."""
    try:
        query = db.query(models.ChatHistory).filter(models.ChatHistory.chat_id == chat_id)
        
        if time_window is not None:
            cutoff_date = datetime.now() - timedelta(days=time_window)
            query = query.filter(models.ChatHistory.created_at >= cutoff_date)
        
        # Получаем статистику
        total_messages = query.count()
        answered_messages = query.filter(models.ChatHistory.is_answered == True).count()
        avg_response_time = db.query(
            func.avg(
                func.extract('epoch', models.ChatHistory.updated_at - models.ChatHistory.created_at)
            )
        ).filter(
            models.ChatHistory.is_answered == True
        ).scalar() or 0
        
        return {
            'total_messages': total_messages,
            'answered_messages': answered_messages,
            'unanswered_messages': total_messages - answered_messages,
            'response_rate': (answered_messages / total_messages * 100) if total_messages > 0 else 0,
            'avg_response_time_seconds': float(avg_response_time),
        }
    except Exception as e:
        logger.error(f"Ошибка при анализе истории чата {chat_id}: {e}", exc_info=True)
        return {
            'error': str(e),
            'total_messages': 0,
            'answered_messages': 0,
            'unanswered_messages': 0,
            'response_rate': 0,
            'avg_response_time_seconds': 0,
        }
