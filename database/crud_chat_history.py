import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from sqlalchemy.orm import Session
from sqlalchemy import select, update as sql_update, delete as sql_delete, func
from sqlalchemy.exc import SQLAlchemyError
from pgvector.sqlalchemy import Vector

from . import models
from .embeddings import get_embeddings

logger = logging.getLogger(__name__)

# Функции, перенесенные из crud.py, относящиеся к ChatHistory

def add_chat_history(
    db: Session,
    chat_id: int,
    user_id: int,
    message_text: str,
    response_text: Optional[str] = None,
    is_answered: bool = False
) -> Optional[models.ChatHistory]:
    """Добавляет запись в историю чата с эмбеддингом."""
    try:
        embeddings_list = get_embeddings([message_text])
        embedding = embeddings_list[0] if embeddings_list else None

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
    """Ищет записи в истории чата по векторному сходству."""
    try:
        query_embeddings_list = get_embeddings([query])
        query_embedding = query_embeddings_list[0] if query_embeddings_list else None

        if query_embedding is None:
            logger.error(f"Не удалось получить эмбеддинг для запроса истории чата: '{query[:100]}...'")
            return []

        max_distance = 1.0 - min_similarity

        # Начинаем строить запрос
        stmt = select(models.ChatHistory).where(models.ChatHistory.embedding != None)

        # Добавляем условие векторного поиска
        stmt = stmt.where(models.ChatHistory.embedding.cosine_distance(query_embedding) <= max_distance)

        # Добавляем фильтр по chat_id, если он указан
        if chat_id is not None:
            stmt = stmt.where(models.ChatHistory.chat_id == chat_id)

        # Добавляем фильтр по времени, если он указан
        if time_window_days is not None and time_window_days > 0:
            cutoff_date = datetime.now() - timedelta(days=time_window_days)
            stmt = stmt.where(models.ChatHistory.created_at >= cutoff_date)

        # Добавляем сортировку и лимит
        stmt = (
            stmt.order_by(models.ChatHistory.embedding.cosine_distance(query_embedding))
            .limit(limit)
        )

        results = db.execute(stmt).scalars().all()
        search_scope = f"в чате {chat_id}" if chat_id else "во всех чатах"
        time_scope = f" за последние {time_window_days} дней" if time_window_days else ""
        logger.info(f"Найдено {len(results)} записей в истории чата {search_scope}{time_scope} для '{query[:50]}...' (порог {min_similarity})" )
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
    """Обновляет ответ и статус для записи в истории чата."""
    try:
        values_to_update: Dict[str, Any] = {
            "response_text": response_text,
            "is_answered": is_answered,
            "updated_at": func.now() # Обновляем время последнего изменения
        }
        if answer_quality is not None:
            values_to_update["answer_quality"] = answer_quality

        # Используем update и returning для получения обновленного объекта (если СУБД поддерживает)
        stmt = (
            sql_update(models.ChatHistory)
            .where(models.ChatHistory.id == entry_id)
            .values(**values_to_update)
            .returning(models.ChatHistory) # Возвращаем обновленные поля
        )

        # Выполняем и получаем результат
        result = db.execute(stmt).scalar_one_or_none()

        if result:
            db.commit() # Коммитим изменения
            logger.info(f"Обновлен ответ для записи истории чата ID={entry_id}")
            # result уже содержит обновленный объект
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
    """Пакетно добавляет записи в историю чата."""
    results: List[Optional[models.ChatHistory]] = []
    if not entries:
        return results

    # Получаем эмбеддинги для всех сообщений одним запросом
    messages = [entry.get('message_text', '') for entry in entries]
    embeddings_list = get_embeddings(messages) if messages else []

    if embeddings_list is None:
        logger.error("Ошибка получения эмбеддингов для пакетного добавления истории чата. Добавление без эмбеддингов.")
        embeddings_map = {}
    elif len(embeddings_list) != len(messages):
        logger.error(f"Количество эмбеддингов ({len(embeddings_list)}) не совпадает с количеством сообщений ({len(messages)}) для пакетного добавления.")
        # Создаем пустую карту или карту с None, чтобы избежать ошибок ниже
        embeddings_map = {msg: None for msg in messages}
    else:
        # Создаем словарь {текст_сообщения: эмбеддинг}
        embeddings_map = dict(zip(messages, embeddings_list))

    db_entries_to_add = []
    original_indices = [] # Храним исходные индексы для сопоставления результатов
    for i, entry in enumerate(entries):
        message_text = entry.get('message_text', '')
        # Получаем эмбеддинг из карты
        embedding = embeddings_map.get(message_text)

        db_entry = models.ChatHistory(
            chat_id=entry.get('chat_id'),
            user_id=entry.get('user_id'),
            message_text=message_text,
            response_text=entry.get('response_text'),
            embedding=embedding,
            is_answered=entry.get('is_answered', (entry.get('response_text') is not None)),
            # created_at и updated_at устанавливаются по умолчанию
        )
        db_entries_to_add.append(db_entry)
        original_indices.append(i)

    if not db_entries_to_add:
        logger.warning("Нет записей для пакетного добавления в историю чата.")
        return [None] * len(entries) # Возвращаем список None правильной длины

    try:
        # Добавляем все подготовленные объекты
        db.add_all(db_entries_to_add)
        # Коммитим транзакцию
        db.commit()

        # После коммита получаем ID и другие сгенерированные значения
        # Важно: db.refresh() не работает с add_all напрямую.
        # Нам нужно вернуть добавленные объекты. Если СУБД поддерживает RETURNING,
        # ID могли бы быть получены сразу, но для универсальности вернем объекты из списка.
        # Объекты в db_entries_to_add должны обновиться после commit (если сессия не закрыта).

        # Создаем итоговый список с None на случай ошибок
        final_results: List[Optional[models.ChatHistory]] = [None] * len(entries)
        for i, entry_obj in enumerate(db_entries_to_add):
            original_index = original_indices[i]
            # Проверяем, присвоен ли ID после коммита
            if entry_obj.id is not None:
                final_results[original_index] = entry_obj
            else:
                # Если ID не присвоен (очень странно), логируем ошибку
                logger.error(f"Объект ChatHistory не получил ID после пакетного добавления (исходный индекс {original_index})")

        successful_count = sum(1 for r in final_results if r is not None)
        logger.info(f"Пакетно добавлено {successful_count} из {len(entries)} записей в историю чата.")
        return final_results

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
    """Получает последние записи истории для указанного чата."""
    try:
        stmt = (
            select(models.ChatHistory)
            .where(models.ChatHistory.chat_id == chat_id)
            .order_by(models.ChatHistory.created_at.desc()) # Сортируем по убыванию даты
            .offset(offset)
            .limit(limit)
        )
        result = db.execute(stmt).scalars().all()
        # Возвращаем в хронологическом порядке (самые старые первыми), если нужно для контекста
        # return result[::-1]
        return result # Пока возвращаем как есть (самые новые первыми)
    except SQLAlchemyError as e:
        logger.error(f"Ошибка SQLAlchemy при получении истории чата {chat_id}: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Неожиданная ошибка при получении истории чата {chat_id}: {e}", exc_info=True)
        return []

def cleanup_chat_history(
    db: Session,
    days: int = 30,
    batch_size: int = 1000 # Размер пакета для удаления
) -> int:
    """Удаляет старые записи истории чата пакетами."""
    if days <= 0:
        logger.warning("Количество дней для очистки истории чата должно быть > 0.")
        return 0
    if batch_size <= 0:
        logger.warning("Размер пакета для очистки истории чата должен быть > 0.")
        batch_size = 1000 # Значение по умолчанию, если указан некорректный

    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        total_deleted = 0
        logger.info(f"Начало очистки истории чата старше {cutoff_date.strftime('%Y-%m-%d')} пакетами по {batch_size}...")

        while True:
            # Выбираем ID записей для удаления в текущем пакете
            ids_to_delete_stmt = (
                select(models.ChatHistory.id)
                .where(models.ChatHistory.created_at < cutoff_date)
                .limit(batch_size)
            )
            ids_result = db.execute(ids_to_delete_stmt).scalars().all()

            if not ids_result:
                logger.info("Больше нет записей истории чата для удаления.")
                break # Выходим из цикла, если удалять больше нечего

            # Удаляем выбранные записи
            delete_stmt = sql_delete(models.ChatHistory).where(models.ChatHistory.id.in_(ids_result))
            result = db.execute(delete_stmt)
            deleted_in_batch = result.rowcount

            # Коммитим транзакцию для текущего пакета
            try:
                db.commit()
                total_deleted += deleted_in_batch
                logger.info(f"Удалено {deleted_in_batch} записей истории чата в пакете. Всего: {total_deleted}.")
            except SQLAlchemyError as commit_err:
                logger.error(f"Ошибка SQLAlchemy при коммите удаления пакета истории чата: {commit_err}", exc_info=True)
                db.rollback()
                logger.warning("Откат транзакции удаления пакета. Остановка очистки.")
                # Возвращаем количество уже удаленных записей
                return total_deleted

            # Если удалено меньше, чем размер пакета, значит, это был последний пакет
            if deleted_in_batch < batch_size:
                break

        logger.info(f"Завершена очистка истории чата старше {days} дней. Всего удалено: {total_deleted}.")
        return total_deleted
    except SQLAlchemyError as e:
        logger.error(f"Ошибка SQLAlchemy при очистке истории чата: {e}", exc_info=True)
        db.rollback()
        return -1 # Возвращаем -1 в случае ошибки
    except Exception as e:
        logger.error(f"Неожиданная ошибка при очистке истории чата: {e}", exc_info=True)
        db.rollback()
        return -1

def analyze_chat_history(
    db: Session,
    chat_id: int,
    time_window_days: Optional[int] = None
) -> Dict[str, Any]:
    """Анализирует статистику по истории конкретного чата."""
    try:
        # Базовый запрос для выборки из нужного чата
        base_stmt = select(models.ChatHistory).where(models.ChatHistory.chat_id == chat_id)

        # Добавляем фильтр по времени, если указан
        if time_window_days is not None and time_window_days > 0:
            cutoff_date = datetime.now() - timedelta(days=time_window_days)
            base_stmt = base_stmt.where(models.ChatHistory.created_at >= cutoff_date)

        # Создаем CTE (Common Table Expression) или подзапрос для удобства
        subq = base_stmt.subquery()

        # Считаем общее количество сообщений
        total_messages_stmt = select(func.count()).select_from(subq)
        total_messages = db.execute(total_messages_stmt).scalar() or 0

        # Считаем количество отвеченных сообщений
        answered_stmt = select(func.count()).select_from(subq).where(subq.c.is_answered == True)
        answered_messages = db.execute(answered_stmt).scalar() or 0

        # Считаем среднее время ответа (если есть отвеченные)
        avg_response_time = 0.0
        if answered_messages > 0:
            # Разница между updated_at (время ответа) и created_at (время вопроса)
            # Используем func.extract('epoch', ...) для получения разницы в секундах
            avg_response_time_stmt = select(
                func.avg(
                    func.extract('epoch', subq.c.updated_at - subq.c.created_at)
                )
            ).select_from(subq).where(subq.c.is_answered == True)
            avg_response_time = db.execute(avg_response_time_stmt).scalar() or 0.0

        return {
            'chat_id': chat_id,
            'time_window_days': time_window_days,
            'total_messages': total_messages,
            'answered_messages': answered_messages,
            'unanswered_messages': total_messages - answered_messages,
            'response_rate': (answered_messages / total_messages * 100.0) if total_messages > 0 else 0.0,
            'avg_response_time_seconds': float(avg_response_time), # Преобразуем в float
        }
    except SQLAlchemyError as e:
        logger.error(f"Ошибка SQLAlchemy при анализе истории чата {chat_id}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Неожиданная ошибка при анализе истории чата {chat_id}: {e}", exc_info=True)

    # Возвращаем словарь с ошибкой или нулями в случае проблем
    return {
        'error': "Произошла ошибка при анализе истории чата",
        'chat_id': chat_id,
        'time_window_days': time_window_days,
        'total_messages': 0,
        'answered_messages': 0,
        'unanswered_messages': 0,
        'response_rate': 0.0,
        'avg_response_time_seconds': 0.0,
    } 