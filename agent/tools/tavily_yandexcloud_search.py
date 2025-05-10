import logging
import os
import json
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
import requests

logger = logging.getLogger(__name__)


class TavilyYandexCloudSearchInput(BaseModel):
    query: str = Field(
        description="Вопрос пользователя для поиска по документации Yandex Cloud."
    )


class TavilyYandexCloudSearchTool(BaseTool):
    name: str = "tavily_yandexcloud_search"
    description: str = (
        "Ищет релевантную информацию только по документации Yandex Cloud (https://yandex.cloud/ru/docs) "
        "и всем её подпапкам через Tavily API. Используй для любых вопросов по Yandex Cloud."
    )
    args_schema: Type[BaseModel] = TavilyYandexCloudSearchInput

    def _run(
        self, query: str, run_manager: CallbackManagerForToolRun | None = None
    ) -> str:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            logger.error("TAVILY_API_KEY не найден в переменных окружения.")
            return "Ошибка: API ключ Tavily не настроен."

        tavily_url = "https://api.tavily.com/search"
        data = {
            "api_key": api_key,
            "query": query,
            "include_domains": ["yandex.cloud/ru/docs"],
            "max_results": 3,
            "search_depth": "advanced",
        }
        logger.debug(
            f"Отправка запроса в Tavily API. URL: {tavily_url}, Data: {json.dumps(data, ensure_ascii=False)}"
        )

        try:
            resp = requests.post(tavily_url, json=data, timeout=20)
            logger.debug(f"Tavily API Response Status: {resp.status_code}")
            logger.debug(f"Tavily API Response Headers: {resp.headers}")
            logger.debug(f"Tavily API Response Body (raw): {resp.text[:1000]}")

            resp.raise_for_status()

            results_data = resp.json()
            logger.debug(f"Tavily API Response JSON: {results_data}")

            results = results_data.get("results", [])
            if not results:
                tavily_answer = results_data.get("answer")
                if tavily_answer:
                    logger.info(
                        f"Tavily не нашел конкретных источников, но дал прямой ответ: {tavily_answer}"
                    )
                    return f"Tavily AI ответил: {tavily_answer}"
                return (
                    "По вашему запросу ничего не найдено в документации Yandex Cloud."
                )

            answer = "Вот релевантные фрагменты из документации Yandex Cloud:\n"
            for r in results:
                snippet = r.get("content", "")[:500]
                url = r.get("url", "")
                title = r.get("title", "")
                answer += f"---\nЗаголовок: {title}\nФрагмент: {snippet}...\nИсточник: {url}\n"
            return answer
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"Tavily API HTTP error: {http_err}", exc_info=False)
            logger.error(
                f"Tavily API Response status code: {http_err.response.status_code}"
            )
            try:
                error_details = http_err.response.json()
                logger.error(f"Tavily API error details (JSON): {error_details}")
                return f"Ошибка поиска по Tavily (HTTP {http_err.response.status_code}): {error_details.get('error', str(http_err))}"
            except json.JSONDecodeError:
                logger.error(
                    f"Tavily API error response (text): {http_err.response.text}"
                )
                return f"Ошибка поиска по Tavily (HTTP {http_err.response.status_code}): {http_err.response.text}"
        except Exception as e:
            logger.error(
                f"Неожиданная ошибка при работе с Tavily API: {e}", exc_info=True
            )
            return f"Неожиданная ошибка при поиске по Tavily: {e}"
