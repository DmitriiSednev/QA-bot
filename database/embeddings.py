import logging
from typing import Optional, List
import os
from functools import lru_cache
import numpy as np

from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

# Инициализация модели эмбеддингов
embeddings_model = OpenAIEmbeddings(
    openai_api_key=os.getenv("API_KEY"),
    openai_api_base=os.getenv("API_BASE")
)

@lru_cache(maxsize=1000)
def get_embeddings(text: str) -> Optional[List[float]]:
    """Получает эмбеддинги для текста."""
    try:
        # TODO: Реализовать получение эмбеддингов через API
        # Временная заглушка
        return [0.0] * 1536
    except Exception as e:
        logger.error(f"Ошибка при получении эмбеддингов: {e}")
        return None

def clear_embedding_cache():
    """Очищает кэш эмбеддингов."""
    get_embeddings.cache_clear() 