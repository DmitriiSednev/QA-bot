import os
import logging
from contextlib import contextmanager
from typing import Optional, Generator, Any, Union, ContextManager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    logger.warning(
        "DATABASE_URL не установлена. Работа с базой данных будет невозможна."
        " Убедитесь, что переменная задана в .env файле."
    )
    # Установка Engine в None, чтобы можно было проверить перед использованием
    engine = None
    SessionLocal = None
else:
    try:
        # echo=True полезно для отладки SQL запросов
        engine = create_engine(DATABASE_URL, echo=False)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        logger.info("Подключение к базе данных успешно настроено.")
    except SQLAlchemyError as e:
        logger.error(f"Ошибка при создании подключения к БД: {e}", exc_info=True)
        engine = None
        SessionLocal = None
    except Exception as e:  # Ловим другие возможные ошибки парсинга URL и т.д.
        logger.error(f"Неожиданная ошибка при настройке БД: {e}", exc_info=True)
        engine = None
        SessionLocal = None


@contextmanager
def get_db_session() -> Generator[Optional[Session], None, None]:
    """Предоставляет сессию SQLAlchemy для работы с БД.

    Используется как менеджер контекста:
        with get_db_session() as db:
            # работа с db (сессией)
            if db:
                ...
            else:
                # Обработка случая, когда сессия не создана
                ...

    Гарантирует закрытие сессии после использования.
    Возвращает None, если SessionLocal не был инициализирован.
    """
    if not SessionLocal:
        logger.error("Попытка получить сессию БД, но SessionLocal не инициализирован.")
        yield None
        return

    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        logger.error(f"Ошибка SQLAlchemy в сессии: {e}", exc_info=True)
        db.rollback()  # Откатываем транзакцию при ошибке
        raise  # Перевыбрасываем ошибку для обработки выше
    except Exception as e:
        logger.error(f"Неожиданная ошибка в сессии БД: {e}", exc_info=True)
        db.rollback()
        raise
    finally:
        db.close()


# Дополнительная функция для проверки доступности БД (опционально)
def check_db_connection() -> bool:
    """Проверяет, можно ли установить соединение с БД."""
    if not engine:
        logger.warning("Проверка соединения: Engine не инициализирован.")
        return False
    try:
        with engine.connect() as connection:
            logger.info("Проверка соединения с БД: Успешно.")
            return True
    except SQLAlchemyError as e:
        logger.error(f"Проверка соединения с БД: Ошибка - {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Проверка соединения с БД: Неожиданная ошибка - {e}", exc_info=True)
        return False
