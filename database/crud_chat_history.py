import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from sqlalchemy.orm import Session
from sqlalchemy import select, update as sql_update, delete as sql_delete, func
from sqlalchemy.exc import SQLAlchemyError
# from pgvector.sqlalchemy import Vector # Закомментировано, если pgvector не используется или вызывает проблемы

from . import models
# Заменяем get_embeddings на get_embedding если функция ожидает один текст
# или обрабатываем список внутри функций
from .embeddings import get_embeddings # Предполагаем, что get_embeddings может принять список текстов

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
        # get_embeddings ожидает список строк и возвращает список списков float или None
        embeddings_result = get_embeddings([message_text])
        embedding = embeddings_result[0] if embeddings_result and embeddings_result[0] else None

        if embedding is None:
            logger.warning(f"Не удалось получить эмбеддинг для сообщения истории чата: '{message_text[:100]}...'. Запись будет добавлена БЕЗ эмбеддинга.")

        db_entry = models.ChatHistory(
            chat_id=chat_id,
            user_id=user_id,
            message_text=message_text,
            response_text=response_text,
            embedding=embedding, # embedding может быть None
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
        query_embedding = query_embeddings_list[0] if query_embeddings_list and query_embeddings_list[0] else None

        if query_embedding is None:
            logger.error(f"Не удалось получить эмбеддинг для запроса истории чата: '{query[:100]}...'")
            return []

        max_distance = 1.0 - min_similarity

        # Начинаем строить запрос
        stmt = select(models.ChatHistory).where(models.ChatHistory.embedding != None)

        # Добавляем условие векторного поиска
        # Убедимся, что тип query_embedding совместим с .cosine_distance
        # pgvector ожидает list[float] или numpy.ndarray
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
    except Exception as e: # Более общая ошибка для проблем с pgvector
        if "operator does not exist: vector" in str(e).lower():
            logger.error(
                f"Ошибка векторного поиска ChatHistory: оператор не найден. Расширение pgvector включено и совместимо с типом эмбеддинга? Ошибка: {e}",
                exc_info=True,
            )
        else:
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
    """Пакетно добавляет записи в историю чата."""
    results: List[Optional[models.ChatHistory]] = [None] * len(entries) # Инициализируем список для результатов
    if not entries:
        return results

    messages_texts = [entry.get('message_text', '') for entry in entries]
    
    all_embeddings_list = get_embeddings(messages_texts) if messages_texts else []

    if all_embeddings_list is None:
        logger.error("Ошибка получения эмбеддингов для пакетного добавления истории чата. Добавление без эмбеддингов.")
        # В этом случае все эмбеддинги будут None
        embeddings_map = {text: None for text in messages_texts}
    elif len(all_embeddings_list) != len(messages_texts):
        logger.error(f"Количество эмбеддингов ({len(all_embeddings_list)}) не совпадает с количеством сообщений ({len(messages_texts)}) для пакетного добавления.")
        embeddings_map = {text: None for text in messages_texts} # Все эмбеддинги None
    else:
        embeddings_map = dict(zip(messages_texts, all_embeddings_list))

    db_entries_to_add = []
    original_indices_map = {} # Для сопоставления объектов с исходными индексами после commit

    for i, entry_data in enumerate(entries):
        message_text = entry_data.get('message_text', '')
        embedding = embeddings_map.get(message_text) # Может быть None

        db_entry_obj = models.ChatHistory(
            chat_id=entry_data.get('chat_id'),
            user_id=entry_data.get('user_id'),
            message_text=message_text,
            response_text=entry_data.get('response_text'),
            embedding=embedding,
            is_answered=entry_data.get('is_answered', (entry_data.get('response_text') is not None)),
        )
        db_entries_to_add.append(db_entry_obj)
        original_indices_map[id(db_entry_obj)] = i # Сохраняем индекс по id объекта в памяти

    if not db_entries_to_add:
        logger.warning("Нет записей для пакетного добавления в историю чата.")
        return results

    try:
        db.add_all(db_entries_to_add)
        db.commit()

        # Обновляем результаты на основе добавленных объектов
        successful_count = 0
        for db_obj in db_entries_to_add:
            if db_obj.id is not None: # Проверяем, что объект был успешно сохранен и получил ID
                original_idx = original_indices_map.get(id(db_obj))
                if original_idx is not None:
                    results[original_idx] = db_obj
                    successful_count +=1
                else: # Очень маловероятно
                    logger.error(f"Не удалось найти исходный индекс для объекта ChatHistory с ID {db_obj.id} после batch add.")
            else: # Если объект не получил ID
                 logger.error(f"Объект ChatHistory не получил ID после пакетного добавления. Данные: { {k:v for k,v in db_obj.__dict__.items() if not k.startswith('_')} }")


        logger.info(f"Пакетно добавлено {successful_count} из {len(entries)} записей в историю чата.")
        return results

    except SQLAlchemyError as e:
        logger.error(f"Ошибка SQLAlchemy при пакетном добавлении записей в историю чата: {e}", exc_info=True)
        db.rollback()
        return [None] * len(entries) # Возвращаем список None той же длины
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
            .order_by(models.ChatHistory.created_at.desc()) 
            .offset(offset)
            .limit(limit)
        )
        result = db.execute(stmt).scalars().all()
        return result 
    except SQLAlchemyError as e:
        logger.error(f"Ошибка SQLAlchemy при получении истории чата {chat_id}: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Неожиданная ошибка при получении истории чата {chat_id}: {e}", exc_info=True)
        return []

def cleanup_chat_history(
    db: Session,
    days: int = 30,
    batch_size: int = 1000 
) -> int:
    """Удаляет старые записи истории чата пакетами."""
    if days <= 0:
        logger.warning("Количество дней для очистки истории чата должно быть > 0.")
        return 0
    if batch_size <= 0:
        logger.warning("Размер пакета для очистки истории чата должен быть > 0.")
        batch_size = 1000 

    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        total_deleted = 0
        logger.info(f"Начало очистки истории чата старше {cutoff_date.strftime('%Y-%m-%d')} пакетами по {batch_size}...")

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
    """Анализирует статистику по истории конкретного чата."""
    try:
        base_stmt = select(models.ChatHistory).where(models.ChatHistory.chat_id == chat_id)

        if time_window_days is not None and time_window_days > 0:
            cutoff_date = datetime.now() - timedelta(days=time_window_days)
            base_stmt = base_stmt.where(models.ChatHistory.created_at >= cutoff_date)

        subq = base_stmt.subquery()

        total_messages_stmt = select(func.count()).select_from(subq)
        total_messages = db.execute(total_messages_stmt).scalar() or 0

        answered_stmt = select(func.count()).select_from(subq).where(subq.c.is_answered == True)
        answered_messages = db.execute(answered_stmt).scalar() or 0

        avg_response_time = 0.0
        if answered_messages > 0:
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
            'avg_response_time_seconds': float(avg_response_time), 
        }
    except SQLAlchemyError as e:
        logger.error(f"Ошибка SQLAlchemy при анализе истории чата {chat_id}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Неожиданная ошибка при анализе истории чата {chat_id}: {e}", exc_info=True)

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

# Код из old-main (заглушки) был удален, так как мы выбрали HEAD.
# Если здесь были какие-то уникальные комментарии или структура из old-main,
# которые нужно было бы сохранить, их пришлось бы переносить вручную.
# В данном случае, версия из HEAD является более полной реализацией.
