import logging
import os
from langchain_openai import ChatOpenAI

# Попытка импортировать YandexGPT для type hinting или будущих нужд,
# но основная работа будет через ChatOpenAI с прокси.
try:
    from langchain_community.chat_models.yandex import ChatYandexGPT

    YANDEX_AVAILABLE = True
except ImportError:
    YANDEX_AVAILABLE = False
    ChatYandexGPT = None  # для type hinting

logger = logging.getLogger(__name__)

# Глобальная переменная для LLM, чтобы избежать многократной инициализации
_llm_instance = None


def setup_llm():
    """Инициализирует и возвращает LLM для использования в агенте.
    Использует ChatOpenAI для подключения к прокси (например, LiteLLM),
    который в свою очередь может быть настроен на YandexGPT.
    """
    global _llm_instance
    if _llm_instance is not None:
        logger.debug("Возвращаем существующий экземпляр LLM.")
        return _llm_instance

    logger.info("Инициализация LLM...")

    # Используем переменные, которые вы указали для .env
    # OPENAI_BASE_URL - это адрес вашего LiteLLM прокси
    # OPENAI_API_KEY - это ключ для вашего LiteLLM прокси (может быть фиктивным, если прокси не требует)
    llm_proxy_base_url = os.getenv("OPENAI_BASE_URL")
    llm_proxy_api_key = os.getenv("OPENAI_API_KEY")
    llm_model_name = os.getenv(
        "LLM_MODEL", "yandexgpt"
    )  # Модель по умолчанию, если не указана

    if not llm_proxy_base_url:
        logger.error(
            "OPENAI_BASE_URL (адрес прокси LiteLLM) не найден в переменных окружения. "
            "LLM не может быть инициализирована."
        )
        return None

    if not llm_proxy_api_key:
        logger.warning(
            "OPENAI_API_KEY (ключ для прокси LiteLLM) не найден. "
            "Для некоторых моделей/настроек прокси это может быть необходимо. "
            "Используется фиктивный ключ 'EMPTY'."
        )
        llm_proxy_api_key = "EMPTY"  # LiteLLM часто работает с фиктивным ключом

    logger.info(f"LLM Proxy: Модель '{llm_model_name}' через '{llm_proxy_base_url}'")

    try:
        _llm_instance = ChatOpenAI(
            model=llm_model_name,  # Имя модели, которое понимает ваш LiteLLM прокси
            openai_api_base=llm_proxy_base_url,
            openai_api_key=llm_proxy_api_key,
            # Можно добавить другие параметры ChatOpenAI, если нужно, например, timeout
            # request_timeout=60,
        )
        logger.info(
            f"LLM (ChatOpenAI через прокси) успешно инициализирована для модели '{llm_model_name}'."
        )
        return _llm_instance
    except Exception as e:
        logger.error(
            f"Ошибка при инициализации ChatOpenAI через прокси: {e}", exc_info=True
        )
        _llm_instance = None
        return None


# Код ниже не нужен, так как мы всегда используем ChatOpenAI с прокси
# и не пытаемся инициализировать YandexGPT напрямую из этого модуля.

# if __name__ == "__main__":
#     # Пример использования и проверки
#     from dotenv import load_dotenv
#     load_dotenv()
#     logging.basicConfig(level=logging.INFO)
#     llm = setup_llm()
#     if llm:
#         print("LLM успешно настроена.")
#         # test_prompt = "Привет! Как дела?"
#         # try:
#         #     response = llm.invoke(test_prompt)
#         #     print(f"Ответ от LLM на '{test_prompt}': {response}")
#         # except Exception as e:
#         #     print(f"Ошибка при вызове LLM: {e}")
#     else:
#         print("Не удалось настроить LLM.")
