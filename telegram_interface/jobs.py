import logging
import os
from datetime import timedelta

# --- Telegram Imports ---
from telegram.ext import ContextTypes, Application

# --- Database Imports ---
from database import connection
from database.crud_faq import cleanup_old_faq_entries
from database.crud_chat_history import cleanup_chat_history

logger = logging.getLogger(__name__)


async def cleanup_scheduler(context: ContextTypes.DEFAULT_TYPE):
    """Периодически запускает очистку старых записей FAQ и истории чатов."""
    logger.info("Запуск периодической очистки данных...")
    # Получаем значения из bot_data, установленные в main.py
    application = context.application
    if not application:
        logger.error("Не удалось получить application из context в cleanup_scheduler.")
        return

    days_to_keep_faq = application.bot_data.get("days_to_keep_faq", 365)
    days_to_keep_history = application.bot_data.get("days_to_keep_history", 30)

    try:
        with connection.get_db_session() as db:
            if not db:
                logger.error("Не удалось получить сессию БД для очистки.")
                return
            # Очистка FAQ
            deleted_faq = cleanup_old_faq_entries(db, days=days_to_keep_faq)
            logger.info(
                f"Очистка FAQ: удалено {deleted_faq} записей старше {days_to_keep_faq} дней."
            )

            # Очистка истории чатов (вызов функции из crud_chat_history)
            try:
                deleted_history = cleanup_chat_history(db, days=days_to_keep_history)
                logger.info(
                    f"Очистка истории чатов: удалено {deleted_history} записей старше {days_to_keep_history} дней."
                )
            except Exception as history_err:
                logger.error(
                    f"Ошибка при очистке истории чатов: {history_err}", exc_info=True
                )

    except Exception as e:
        logger.error(f"Ошибка во время периодической очистки: {e}", exc_info=True)


# Новая функция для настройки задач
def setup_jobs(application: Application):
    """Настраивает и запускает периодические задачи."""
    job_queue = application.job_queue
    cleanup_interval_hours = int(os.getenv("CLEANUP_INTERVAL_HOURS", "24"))

    if cleanup_interval_hours > 0:
        job_queue.run_repeating(
            cleanup_scheduler,
            interval=timedelta(hours=cleanup_interval_hours),
            first=timedelta(minutes=5),
            name="periodic_cleanup",
        )
        logger.info(
            f"Запланирована периодическая очистка данных (интервал {cleanup_interval_hours}ч)."
        )
    else:
        logger.info(
            "Периодическая очистка данных отключена (CLEANUP_INTERVAL_HOURS <= 0)."
        )
    # Добавь здесь настройку других задач, если они есть
