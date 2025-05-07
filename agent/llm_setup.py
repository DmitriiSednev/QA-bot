import logging
import os
from typing import Optional

from langchain_community.chat_models import ChatYandexGPT

logger = logging.getLogger(__name__)


def setup_llm() -> Optional[ChatYandexGPT]:
    """Инициализирует и возвращает модель ЛЛМ."""
    yandex_api_key = os.getenv("YANDEX_API_KEY")
    yandex_folder_id = os.getenv("YANDEX_FOLDER_ID")
    llm_model_uri = os.getenv("LLM_MODEL")

    if not all([yandex_api_key, yandex_folder_id, llm_model_uri]):
        logger.critical(
            "Переменные окружения YANDEX_API_KEY, YANDEX_FOLDER_ID или LLM_MODEL не установлены. "
            "Агент не может быть инициализирован."
        )
        return None

    try:
        # Используем асинхронную версию для совместимости с узлами графа
        llm = ChatYandexGPT(
            api_key=yandex_api_key,
            folder_id=yandex_folder_id,
            model_uri=llm_model_uri,
            temperature=0.6,  # Можно настроить
            max_tokens=1500,  # Можно настроить
        )
        logger.info(f"Модель ChatYandexGPT ({llm_model_uri}) успешно инициализирована.")
        return llm
    except Exception as e:
        logger.critical(f"Ошибка инициализации ChatYandexGPT: {e}", exc_info=True)
        return None
