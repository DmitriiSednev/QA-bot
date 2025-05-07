import logging
import os
import requests
from typing import Type, Dict, Any, Optional, List

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

# Импортируем базовый класс с кешированием
from .base_search_tool import BaseSearchTool, BaseSearchInput

logger = logging.getLogger(__name__)

# --- Константы ---
YANDEX_SEARCH_API_V2_URL = "https://search-api.yandex.cloud/search/v2/request"
# Документация Yandex Cloud
DEFAULT_YC_DOCS_SITE = "yandex.cloud/ru/docs"


class YandexSearchApiInput(BaseSearchInput):
    """Схема входных данных для YandexSearchApiTool."""

    query: str = Field(description="Текст поискового запроса.")
    site_filter: str = Field(
        default=DEFAULT_YC_DOCS_SITE,
        description="Адрес сайта для ограничения поиска (например, 'yandex.cloud/ru/docs')",
    )
    # folder_id будет браться из окружения, не передаем через аргументы
    # limit и min_similarity наследуются, но min_similarity здесь не используется


class YandexSearchApiTool(BaseSearchTool):
    """Инструмент для поиска информации с использованием Yandex Search API v2 (GenSearch)."""

    name: str = "search_yandex_documentation"
    description: str = (
        "Используй этот инструмент ПЕРВЫМ для поиска ответов на вопросы, касающиеся **Yandex Cloud**, "
        "его сервисов, API, CLI и т.д., используя поиск **по официальной документации** через Yandex Search API. "
        "Возвращает сгенерированный YandexGPT ответ на основе найденной документации."
        "Входными данными должен быть поисковый запрос (вопрос пользователя)."
    )
    args_schema: Type[BaseModel] = YandexSearchApiInput

    def _run(  
        self,
        query: str,
        site_filter: str = DEFAULT_YC_DOCS_SITE,
        limit: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> str:
        """Выполняет поиск через Yandex Search API v2"""
        cache_key_params = {
            "query": query,
            "site_filter": site_filter,
        }  # 2 таба - тело метода
        cache_key = self._get_cache_key(**cache_key_params)
        cached_result = self._get_cached_results(cache_key)
        if cached_result is not None:
            # Ожидаем, что в кеше лежит строка с ответом
            if isinstance(cached_result, str):  
                logger.info(
                    f"Результат для '{query}' (site: {site_filter}) найден в кеше Redis."
                )
                # Вызываем callback об окончании (если есть)
                if run_manager: 
                    run_manager.on_tool_end(cached_result)
                return cached_result
            else:
                logger.warning(
                    f"Найден некорректный кеш для ключа {cache_key}, игнорирую."
                )
        # --- Конец логики кеширования --- #

        logger.info(
            f"Запуск YandexSearchApiTool (не из кеша): query='{query}', site='{site_filter}'"
        )

        yandex_api_key = os.getenv("YANDEX_API_KEY")  
        yandex_folder_id = os.getenv("YANDEX_FOLDER_ID")

        if not yandex_api_key or not yandex_folder_id: 
            msg = "Ключ YANDEX_API_KEY или YANDEX_FOLDER_ID не найдены в .env. Поиск по документации невозможен."
            logger.error(msg)
            return f"Ошибка конфигурации: {msg}"

        headers = {  
            "Authorization": f"Api-Key {yandex_api_key}",
            "Content-Type": "application/json",
        }

        # Тело запроса согласно документации API v2
        # Используем простейший вариант с одним сообщением USER
        request_body = {  
            "folderId": yandex_folder_id,
            "messages": [
                {"role": "ROLE_USER", "content": query}
                # Можно добавить контекст предыдущих сообщений, если нужно
            ],
            "site": {"site": [site_filter]},
            "searchOptions": {
                # "searchType": "SEARCH_TYPE_RUSSIAN_AND_ENGLISH", # Можно указать тип поиска
                # "maxResults": limit # Этот параметр может влиять на количество документов для анализа YandexGPT
            },
            # "enableNrfmDocs": True, # Опция для улучшения качества поиска, может быть полезна
            # "fixMisspell": True # Автоисправление опечаток
        }

        try:  
            logger.debug(f"Отправка запроса в Yandex Search API: {request_body}")
            response = requests.post(
                YANDEX_SEARCH_API_V2_URL, headers=headers, json=request_body, timeout=30
            )
            response.raise_for_status()  # Проверка на HTTP ошибки (4xx, 5xx)

            result_json = response.json()
            logger.debug(f"Ответ от Yandex Search API: {result_json}")

            # Ищем генеративный ответ в результате
            # Путь к ответу может быть таким: result_json['result']['text']
            generative_answer = result_json.get("result", {}).get("text")

            if generative_answer: 
                logger.info("Получен генеративный ответ от Yandex Search API.")
                # Кэшируем успешный ответ (строку)
                self._cache_results(cache_key, generative_answer)
                # Вызываем callback об окончании (если есть)
                if run_manager:  
                    run_manager.on_tool_end(generative_answer)
                return generative_answer
            else:
                error_message = result_json.get("error", {}).get(
                    "message", "Ответ не содержит генеративного текста."
                )
                logger.warning(
                    f"Yandex Search API не вернул генеративный ответ. Детали: {result_json}"
                )
                error_response = f"Поиск по документации ('{site_filter}') не дал результатов по вашему запросу. ({error_message})"
                # Кэшируем сообщение об отсутствии ответа, чтобы не запрашивать снова сразу
                self._cache_results(
                    cache_key, error_response, ttl_seconds=600
                )  # Кэш на 10 минут
                # Вызываем callback об окончании (если есть)
                if run_manager:  
                    run_manager.on_tool_end(error_response)
                return error_response

        except (
            requests.exceptions.RequestException
        ) as e:  
            logger.error(
                f"Ошибка сети при запросе к Yandex Search API: {e}", exc_info=True
            )
            error_response = f"Ошибка сети при поиске в документации: {e}"
            # Не кэшируем сетевые ошибки
            # Вызываем callback об ошибке (если есть)
            if run_manager:  
                run_manager.on_tool_error(e)
            return error_response
        except Exception as e:  
            logger.error(
                f"Неожиданная ошибка при работе с Yandex Search API: {e}", exc_info=True
            )
            error_response = (
                f"Произошла внутренняя ошибка при поиске в документации: {e}"
            )
            # Не кэшируем неожиданные ошибки
            # Вызываем callback об ошибке (если есть)
            if run_manager:  
                run_manager.on_tool_error(e)
            return error_response
