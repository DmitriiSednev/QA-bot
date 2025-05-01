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

# Загрузка переменных окружения
load_dotenv()

# Получение параметров подключения из переменных окружения
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "qa_bot")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASSWORD", "")

# Формирование URL подключения
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Создание движка SQLAlchemy
engine = create_engine(DATABASE_URL)

# Создание фабрики сессий
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Инициализация базы данных и создание таблиц."""
    try:
        # Создаем все таблицы
        models.Base.metadata.create_all(bind=engine)
        
        # Проверяем существование колонки created_at
        with engine.connect() as connection:
            for table in ['faq_entries', 'chat_history', 'admins']:
                try:
                    connection.execute(text(f"SELECT created_at FROM {table} LIMIT 1"))
                except OperationalError:
                    # Если колонки нет, добавляем её
                    connection.execute(text(f"""
                        ALTER TABLE {table} 
                        ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    """))
                    connection.commit()
        
        logger.info("База данных успешно инициализирована")
    except Exception as e:
        logger.error(f"Ошибка при инициализации базы данных: {e}")
        raise

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
