import logging
import os

from dotenv import load_dotenv

# Важно: импортировать Base и engine из connection ДО импорта моделей
from database.connection import engine, check_db_connection

# Импортируем модель, чтобы она была зарегистрирована в Base.metadata
from database.models import Base, FAQEntry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_db():
    # Загружаем переменные окружения (особенно DATABASE_URL)
    load_dotenv()
    logger.info("Загружены переменные окружения...")

    if not engine:
        logger.error("Engine SQLAlchemy не инициализирован. Проверьте DATABASE_URL.")
        return

    logger.info("Проверка соединения с БД...")
    if not check_db_connection():
        logger.error("Не удалось подключиться к базе данных. Таблицы не будут созданы.")
        return

    logger.info(
        f"Создание таблицы '{FAQEntry.__tablename__}' (если она не существует)..."
    )
    try:
        # Создаем все таблицы, определенные в Base.metadata
        Base.metadata.create_all(bind=engine)
        logger.info(
            f"Таблица '{FAQEntry.__tablename__}' успешно создана или уже существует."
        )
    except Exception as e:
        logger.error(f"Ошибка при создании таблицы: {e}", exc_info=True)


if __name__ == "__main__":
    logger.info("Запуск инициализации базы данных...")
    init_db()
    logger.info("Инициализация базы данных завершена.")
