import logging
from datetime import datetime
import asyncio
from database import connection, crud
from typing import Optional

logger = logging.getLogger(__name__)

class Scheduler:
    def __init__(self, interval: int = 24 * 60 * 60):
        self.interval = interval
        self._last_run: Optional[datetime] = None
        self._is_running = False
        self._error_count = 0
        self._max_retries = 3

    async def cleanup_scheduler(self):
        """Планировщик для очистки старых FAQ записей."""
        if self._is_running:
            logger.warning("Планировщик уже запущен")
            return

        self._is_running = True
        while self._is_running:
            try:
                logger.info("Запуск планового удаления старых FAQ записей...")
                with connection.get_db_session() as db:
                    if db:
                        deleted = crud.cleanup_old_faq_entries(db)
                        logger.info(f"Удалено {deleted} старых FAQ записей")
                        self._error_count = 0  # Сброс счетчика ошибок при успехе
                    else:
                        logger.error("Не удалось получить сессию БД для очистки")
                        self._error_count += 1
            except Exception as e:
                logger.error(f"Ошибка при выполнении очистки: {e}", exc_info=True)
                self._error_count += 1
                
                if self._error_count >= self._max_retries:
                    logger.critical("Превышено максимальное количество ошибок. Остановка планировщика.")
                    self._is_running = False
                    break
                
                # Увеличиваем интервал при ошибках
                await asyncio.sleep(min(self.interval * (self._error_count + 1), 24 * 60 * 60))
                continue
            
            self._last_run = datetime.now()
            await asyncio.sleep(self.interval)

    def stop(self):
        """Останавливает планировщик."""
        self._is_running = False

    @property
    def last_run(self) -> Optional[datetime]:
        """Возвращает время последнего запуска."""
        return self._last_run

    @property
    def is_running(self) -> bool:
        """Возвращает статус работы планировщика."""
        return self._is_running 