import redis
import os
import logging
import json
from typing import Optional, Any, Union, Dict, List

logger = logging.getLogger(__name__)

# Загрузка настроек Redis из .env
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)  # Пароль, если есть

# Глобальная переменная для хранения клиента Redis
redis_client = None


class RedisCache:
    """Класс для работы с Redis кешем."""

    def __init__(self, redis_url=None):
        """Инициализирует Redis клиент.

        Args:
            redis_url: URL подключения к Redis (например, redis://host:port/db).
                       Если None, используются переменные REDIS_HOST, REDIS_PORT и т.д.
        """
        self.redis_client = None
        if redis_url:
            try:
                logger.info(f"Подключение к Redis по URL: {redis_url}")
                self.redis_client = redis.from_url(
                    redis_url,
                    decode_responses=False,  # Будем декодировать сами после загрузки JSON
                )
            except Exception as e:
                logger.error(f"Ошибка при подключении к Redis по URL {redis_url}: {e}")
        else:
            self.redis_client = get_redis_client()

    def ping(self) -> bool:
        """Проверяет доступность Redis-сервера."""
        if not self.redis_client:
            return False
        try:
            return self.redis_client.ping()
        except Exception as e:
            logger.error(f"Ошибка при проверке соединения с Redis: {e}")
            return False

    def get_redis_client(self) -> Optional[redis.Redis]:
        """Возвращает клиент Redis."""
        return self.redis_client


def get_redis_client() -> Optional[redis.Redis]:
    """Инициализирует и возвращает клиент Redis."""
    global redis_client
    if redis_client:
        try:
            # Проверяем соединение перед возвратом
            if redis_client.ping():
                return redis_client
            else:
                logger.warning(
                    "Существующее соединение Redis не отвечает. Попытка переподключения."
                )
                redis_client = None  # Сбрасываем для переподключения
        except redis.exceptions.ConnectionError:
            logger.warning(
                "Ошибка проверки существующего соединения Redis. Попытка переподключения."
            )
            redis_client = None  # Сбрасываем для переподключения

    if redis_client is None:
        try:
            logger.info(
                f"Подключение к Redis: host={REDIS_HOST}, port={REDIS_PORT}, db={REDIS_DB}"
            )
            redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=False,  # Будем декодировать сами после загрузки JSON
            )
            redis_client.ping()  # Проверка соединения
            logger.info("Соединение с Redis установлено успешно.")
            return redis_client
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Не удалось подключиться к Redis: {e}")
            redis_client = None  # Убедимся, что клиент сброшен при ошибке
            return None
        except Exception as e:
            logger.error(
                f"Неожиданная ошибка при подключении к Redis: {e}", exc_info=True
            )
            redis_client = None
            return None
    return redis_client  # Возвращаем None если не удалось подключиться


def set_cache(
    key: str, value: Union[Dict, List, str, int, float, bool], ttl_seconds: int = 3600
):
    """Сохраняет значение в кеш Redis с TTL."""
    client = get_redis_client()
    if not client:
        logger.warning("Не удалось сохранить в кеш: Redis недоступен.")
        return

    try:
        # Сериализуем значение в JSON перед сохранением
        serialized_value = json.dumps(value)
        client.setex(key, ttl_seconds, serialized_value)
        logger.debug(
            f"Значение для ключа '{key}' сохранено в кеш Redis с TTL {ttl_seconds} сек."
        )
    except redis.exceptions.RedisError as e:
        logger.error(f"Ошибка Redis при сохранении ключа '{key}': {e}")
    except TypeError as e:
        logger.error(f"Ошибка сериализации JSON при сохранении ключа '{key}': {e}")
    except Exception as e:
        logger.error(
            f"Неожиданная ошибка при сохранении ключа '{key}' в кеш: {e}", exc_info=True
        )


def get_cache(key: str) -> Optional[Any]:
    """Получает значение из кеша Redis."""
    client = get_redis_client()
    if not client:
        logger.warning("Не удалось получить из кеша: Redis недоступен.")
        return None

    try:
        cached_value_bytes = client.get(key)
        if cached_value_bytes:
            # Десериализуем JSON после получения
            cached_value = json.loads(cached_value_bytes.decode("utf-8"))
            logger.debug(f"Значение для ключа '{key}' найдено в кеше Redis.")
            return cached_value
        else:
            logger.debug(f"Ключ '{key}' не найден в кеше Redis.")
            return None
    except redis.exceptions.RedisError as e:
        logger.error(f"Ошибка Redis при получении ключа '{key}': {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка десериализации JSON при получении ключа '{key}': {e}")
        # Возможно, стоит удалить некорректный ключ? client.delete(key)
        return None
    except Exception as e:
        logger.error(
            f"Неожиданная ошибка при получении ключа '{key}' из кеша: {e}",
            exc_info=True,
        )
        return None


def delete_cache(key: str) -> bool:
    """Удаляет значение из кеша Redis."""
    client = get_redis_client()
    if not client:
        logger.warning("Не удалось удалить из кеша: Redis недоступен.")
        return False
    try:
        result = client.delete(key)
        if result > 0:
            logger.debug(f"Ключ '{key}' удален из кеша Redis.")
            return True
        else:
            logger.debug(f"Ключ '{key}' не найден в кеше Redis для удаления.")
            return False
    except redis.exceptions.RedisError as e:
        logger.error(f"Ошибка Redis при удалении ключа '{key}': {e}")
        return False
    except Exception as e:
        logger.error(
            f"Неожиданная ошибка при удалении ключа '{key}' из кеша: {e}", exc_info=True
        )
        return False


def clear_cache_by_prefix(prefix: str) -> int:
    """Очищает все ключи в кеше Redis, начинающиеся с префикса."""
    client = get_redis_client()
    if not client:
        logger.warning("Не удалось очистить кеш по префиксу: Redis недоступен.")
        return 0

    deleted_count = 0
    try:
        # Используем scan для итерации по ключам без блокировки Redis
        for key_bytes in client.scan_iter(f"{prefix}:*"):
            key = key_bytes.decode("utf-8")
            if client.delete(key):
                deleted_count += 1
                logger.debug(f"Удален ключ '{key}' по префиксу '{prefix}'.")
        logger.info(f"Очищено {deleted_count} ключей по префиксу '{prefix}'.")
        return deleted_count
    except redis.exceptions.RedisError as e:
        logger.error(f"Ошибка Redis при очистке кеша по префиксу '{prefix}': {e}")
        return deleted_count  # Возвращаем сколько успели удалить
    except Exception as e:
        logger.error(
            f"Неожиданная ошибка при очистке кеша по префиксу '{prefix}': {e}",
            exc_info=True,
        )
        return deleted_count
