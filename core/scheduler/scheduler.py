import logging
from datetime import datetime
import asyncio
from database import connection, crud

logger = logging.getLogger(__name__)

async def cleanup_scheduler():
    """Планировщик для очистки старых FAQ записей."""
    while True:
        logger.info("Запуск планового удаления старых FAQ записей...")
        try:
            with connection.get_db_session() as db:
                if db:
                    deleted = crud.cleanup_old_faq_entries(db)
                    logger.info(f"Удалено {deleted} старых FAQ записей")
                else:
                    logger.error("Не удалось получить сессию БД для очистки")
        except Exception as e:
            logger.error(f"Ошибка при выполнении очистки: {e}", exc_info=True)
        
        # Ждем 24 часа до следующей проверки
        await asyncio.sleep(24 * 60 * 60)  # 24 часа в секундах 