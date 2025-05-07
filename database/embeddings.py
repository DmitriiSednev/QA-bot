import logging
from typing import Optional, List, Union
import os

# Убираем OpenAIEmbeddings
# from langchain_openai import OpenAIEmbeddings
# Импортируем класс для работы с Yandex API напрямую
import numpy as np
import requests

logger = logging.getLogger(__name__)

# Инициализация ключей API для работы с Yandex
yandex_api_key = os.getenv("YANDEX_API_KEY")
yandex_folder_id = os.getenv("YANDEX_FOLDER_ID")

if not yandex_api_key or not yandex_folder_id:
    logger.warning(
        "YANDEX_API_KEY или YANDEX_FOLDER_ID не найдены в .env. "
        "Работа с эмбеддингами Yandex будет невозможна."
    )
    embeddings_model = None
else:
    try:
        # Вместо использования langchain_community.embeddings.YandexEmbeddings,
        # которого нет в текущей версии библиотеки, реализуем прямой вызов API
        logger.info(
            "Используем встроенную реализацию для обращения к Yandex Embeddings API"
        )
        embeddings_model = "Реализован через прямые вызовы API"
    except Exception as e:
        logger.error(f"Ошибка инициализации Yandex Embeddings: {e}", exc_info=True)
        embeddings_model = None


def get_embeddings(texts: Union[str, List[str]]) -> Optional[List[List[float]]]:
    """Получает эмбеддинги для одного текста или списка текстов через Yandex."""
    if not yandex_api_key or not yandex_folder_id:
        logger.error("Yandex API ключи не настроены. Невозможно получить эмбеддинги.")
        return None

    input_texts: List[str]
    if isinstance(texts, str):
        input_texts = [texts]
        log_msg = f"Получен 1 текст для Yandex эмбеддинга: '{texts[:50]}...'"
    elif isinstance(texts, list):
        input_texts = texts
        log_msg = f"Получено {len(texts)} текстов для Yandex эмбеддинга."
    else:
        logger.error(f"Неверный тип входных данных для get_embeddings: {type(texts)}")
        return None

    if not input_texts:
        logger.debug("Получен пустой список для get_embeddings.")
        return []

    try:
        vectors = []
        for text in input_texts:
            # Прямой вызов Yandex API для получения эмбеддингов
            url = (
                "https://embeddings.api.cloud.yandex.net/embeddings/v1/text-embeddings"
            )
            headers = {
                "Authorization": f"Api-Key {yandex_api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "folderId": yandex_folder_id,
                "inputs": [text],
                "model": "text-search-query",  # или другая модель эмбеддингов
                "outputFormat": "float",
            }

            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

            if "embeddings" in result and result["embeddings"]:
                vectors.append(result["embeddings"][0])
            else:
                logger.error(f"Ошибка в ответе API Yandex: {result}")
                return None

        logger.debug(log_msg + f" -> получено {len(vectors)} векторов.")
        return vectors
    except Exception as e:
        logger.error(f"Ошибка при получении эмбеддингов Yandex: {e}", exc_info=True)
        return None


def clear_embedding_cache():
    """Очищает кэш эмбеддингов (если он будет реализован)."""
    logger.warning(
        "Кэширование на уровне database.embeddings не используется. Вызов clear_embedding_cache не имеет эффекта."
    )
    pass
