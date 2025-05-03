import os
import logging
from contextlib import contextmanager
from typing import Optional, Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from dotenv import load_dotenv
from . import models

# Настройка логирования
logger = logging.getLogger(__name__)

# Загрузка переменных окружения с перезаписью существующих
load_dotenv(override=True)

# Получение параметров подключения из переменных окружения
DB_HOST = os.getenv("DB_HOST") # Убираем значения по умолчанию, т.к. они должны быть в .env
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASSWORD")

# Проверка, что все переменные загружены
if not all([DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS]):
    missing = [var for var in ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"] if not os.getenv(var)]
    logger.critical(f"КРИТИЧЕСКАЯ ОШИБКА: Не найдены переменные окружения для БД: {missing}. Проверьте файл .env")
    # Можно либо выбросить исключение, либо попытаться продолжить с риском ошибки ниже
    raise ValueError(f"Не найдены переменные окружения для БД: {missing}")


# Формирование URL подключения
# Добавим try-except вокруг формирования URL на случай, если порт все еще не число
try:
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{int(DB_PORT)}/{DB_NAME}"
except ValueError as e:
     logger.critical(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось преобразовать DB_PORT ('{DB_PORT}') в число при формировании DATABASE_URL: {e}")
     raise ValueError(f"Неверный формат DB_PORT: {DB_PORT}") from e
except TypeError as e:
     logger.critical(f"КРИТИЧЕСКАЯ ОШИБКА: Одна из переменных для DATABASE_URL не задана (None): {e}")
     # Перепроверяем, что не None перед форматированием
     if not all([DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME]):
         missing = [k for k,v in locals().items() if k.startswith("DB_") and v is None]
         raise ValueError(f"Переменные окружения None при формировании URL: {missing}") from e
     else:
         # Если все переменные есть, но ошибка типа все равно возникла (маловероятно)
         raise TypeError("Ошибка типа при формировании DATABASE_URL, хотя переменные не None") from e


# Создание движка SQLAlchemy
try:
    engine = create_engine(DATABASE_URL)
    # Попытка соединения для ранней проверки
    with engine.connect() as connection_test:
        logger.info("Проверка соединения с БД при инициализации: Успешно.")
except Exception as e:
    logger.critical(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось создать движок SQLAlchemy или подключиться к БД: {e}", exc_info=True)
    # Перевыбрасываем исключение, чтобы приложение не запустилось с нерабочей БД
    raise


# Создание фабрики сессий
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# def init_db():
#     """Инициализация базы данных и создание таблиц."""
#     """(Закомментировано, т.к. схема управляется Alembic)"""
#     try:
#         # Создаем все таблицы
#         models.Base.metadata.create_all(bind=engine)
        
#         # Проверяем существование колонки created_at
#         with engine.connect() as connection:
#             for table in ['faq_entries', 'chat_history', 'admins']:
#                 try:
#                     connection.execute(text(f"SELECT created_at FROM {table} LIMIT 1"))
#                 except OperationalError:
#                     # Если колонки нет, добавляем её
#                     connection.execute(text(f"""
#                         ALTER TABLE {table} 
#                         ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#                     """))
#                     connection.commit()
        
#         logger.info("База данных успешно инициализирована")
#     except Exception as e:
#         logger.error(f"Ошибка при инициализации базы данных: {e}")
#         raise

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Контекстный менеджер для получения сессии базы данных."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class DatabaseConnection:
    """Класс для управления подключением к базе данных."""
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
        logger.info("Подключение к базе данных успешно настроено")
    
    @contextmanager
    def get_db_session(self) -> Generator[Session, None, None]:
        """Получение сессии базы данных."""
        db = self.SessionLocal()
        try:
            yield db
        except SQLAlchemyError as e:
            logger.error(f"Ошибка SQLAlchemy в сессии: {e}", exc_info=True)
            db.rollback()
            raise
        except Exception as e:
            logger.error(f"Неожиданная ошибка в сессии БД: {e}", exc_info=True)
            db.rollback()
            raise
        finally:
            db.close()

    def check_connection(self) -> bool:
        """Проверка соединения с базой данных."""
        try:
            with self.engine.connect():
                logger.info("Проверка соединения с БД: Успешно")
                return True
        except Exception as e:
            logger.error(f"Проверка соединения с БД: Ошибка - {e}", exc_info=True)
            return False

# Создаем глобальный экземпляр подключения
db_connection = DatabaseConnection()
