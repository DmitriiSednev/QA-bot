import logging
import signal
import sys
import portalocker
import os
from typing import Optional

logger = logging.getLogger(__name__)

LOCK_FILE = "/tmp/telegram_bot.lock"  # Убедись, что путь /tmp доступен для записи
lock_handle: Optional[portalocker.Lock] = None


def acquire_lock() -> bool:
    """Пытается захватить блокировку файла, чтобы предотвратить запуск нескольких экземпляров."""
    global lock_handle
    try:
        # Открываем файл и пытаемся получить эксклюзивную блокировку без блокировки
        # Файл будет создан, если не существует
        lock_handle = open(LOCK_FILE, "w")
        portalocker.lock(lock_handle, portalocker.LOCK_EX | portalocker.LOCK_NB)
        logger.info(f"Блокировка файла {LOCK_FILE} успешно установлена.")
        # Записываем PID в файл для информации
        lock_handle.write(str(os.getpid()))
        lock_handle.flush()
        return True
    except portalocker.exceptions.LockException:
        logger.error(
            f"Другой экземпляр бота уже запущен (файл блокировки {LOCK_FILE} занят). Завершение работы."
        )
        if lock_handle:
            lock_handle.close()
            lock_handle = None
        return False
    except Exception as e:
        logger.error(
            f"Не удалось установить блокировку файла {LOCK_FILE}: {e}", exc_info=True
        )
        if lock_handle:
            lock_handle.close()
            lock_handle = None
        return False


def release_lock():
    """Освобождает блокировку файла и удаляет его."""
    global lock_handle
    if lock_handle:
        try:
            portalocker.unlock(lock_handle)
            lock_handle.close()
            lock_handle = None
            logger.info(f"Блокировка файла {LOCK_FILE} снята.")
        except Exception as e:
            logger.error(
                f"Ошибка при снятии блокировки файла {LOCK_FILE}: {e}", exc_info=True
            )
        finally:
            # Пытаемся удалить файл блокировки
            try:
                if os.path.exists(LOCK_FILE):
                    os.remove(LOCK_FILE)
                    logger.info(f"Файл блокировки {LOCK_FILE} удален.")
            except OSError as e:
                logger.error(f"Не удалось удалить файл блокировки {LOCK_FILE}: {e}")


def handle_shutdown(signum, frame):
    """Обрабатывает сигналы завершения (SIGINT, SIGTERM)."""
    logger.warning(
        f"Получен сигнал {signal.Signals(signum).name}. Завершение работы..."
    )
    # Здесь можно добавить логику для грациозного завершения
    # Например, дождаться завершения текущих задач
    release_lock()
    sys.exit(0)


def register_signal_handlers(checkpoint_db_connection=None):
    """Регистрирует обработчики сигналов SIGINT и SIGTERM."""
    # Аргумент checkpoint_db_connection здесь не используется напрямую,
    # так как асинхронное закрытие соединения происходит в finally блоке main.py.
    # Он добавлен для совместимости вызова из main.py.
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    logger.info("Обработчики сигналов SIGINT и SIGTERM зарегистрированы.")
