import logging
import os
from typing import Type, List, Dict, Any, Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults

# Импортируем базовый класс с кешированием, если он есть и используется
# from .base_search_tool import BaseSearchTool, BaseSearchInput
# Если BaseSearchTool не используется или не подходит, можно наследоваться от BaseTool
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

# --- Константы ---
# Домен для поиска по документации Yandex Cloud
YANDEX_CLOUD_DOCS_DOMAIN = "yandex.cloud/ru/docs/"


class TavilyYandexCloudSearchInput(BaseModel):
    """Схема входных данных для TavilyYandexCloudSearchTool."""

    query: str = Field(
        description="Поисковый запрос (вопрос пользователя) для поиска в документации Yandex Cloud."
    )
    # max_results уже есть в TavilySearchResults, можно переопределить или использовать по умолчанию
    # include_domains будет установлен принудительно


class TavilyYandexCloudSearchTool(BaseTool):
    """
    Инструмент для поиска информации по документации Yandex Cloud с использованием Tavily Search API.
    """

    name: str = "tavily_yandexcloud_search"
    description: str = (
        "Используй этот инструмент ПЕРВЫМ для поиска ответов на вопросы, касающиеся **Yandex Cloud**, "
        "его сервисов, API, CLI, устранения неполадок и т.д., "
        "используя поиск **по официальной документации** (`yandex.cloud/ru/docs/`). "
        "Возвращает наиболее релевантные фрагменты из документации."
    )
    args_schema: Type[BaseModel] = TavilyYandexCloudSearchInput

    # Атрибут для хранения инициализированного клиента Tavily
    tavily_search_client: Optional[TavilySearchResults] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            # Ключ TAVILY_API_KEY должен быть в переменных окружения
            self.tavily_search_client = TavilySearchResults(
                max_results=5,  # Количество результатов по умолчанию
                # topic="general", # Можно убрать, если не нужно ограничивать тематику заранее
                # search_depth="advanced", # Можно установить глубину поиска, если нужно
            )
            logger.info("Tavily Search API клиент успешно инициализирован.")
        except Exception as e:
            logger.error(
                f"Ошибка инициализации клиента Tavily Search API: {e}. Убедитесь, что TAVILY_API_KEY установлен.",
                exc_info=True,
            )
            self.tavily_search_client = None

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Выполняет поиск через Tavily Search API, ограничиваясь доменом yandex.cloud/ru/docs.
        """
        if not self.tavily_search_client:
            return "Ошибка: Клиент Tavily Search API не инициализирован. Проверьте TAVILY_API_KEY."

        logger.info(
            f"Запуск TavilyYandexCloudSearchTool с запросом: '{query}' для домена '{YANDEX_CLOUD_DOCS_DOMAIN}'"
        )

        try:
            # Формируем поисковый запрос, чтобы Tavily искал только на указанном сайте
            # Tavily использует параметр 'include_domains' для фильтрации по доменам
            # Для более точного поиска по поддомену можно использовать `site:` в самом запросе,
            # но TavilySearchResults должен корректно обрабатывать include_domains.

            # TavilySearchResults.invoke ожидает словарь аргументов
            # Мы передаем query и указываем include_domains принудительно.
            # Документация Langchain TavilySearch: https://python.langchain.com/docs/integrations/tools/tavily_search/

            tool_input_args = {
                "query": query,
                "include_domains": [YANDEX_CLOUD_DOCS_DOMAIN],
                # "search_depth": "advanced" # можно добавить, если нужно больше деталей
            }

            # Результат будет строкой (JSON по умолчанию) или списком словарей,
            # в зависимости от настроек TavilySearchResults и того, как он обрабатывается BaseTool.
            # Стандартный TavilySearchResults возвращает список словарей.
            # Нам нужно вернуть строку для LLM.

            raw_results: List[Dict[str, Any]] = self.tavily_search_client.invoke(
                tool_input_args
            )

            logger.debug(f"Получены сырые результаты от Tavily: {raw_results}")

            if not raw_results:
                return f"Поиск по документации Yandex Cloud ('{YANDEX_CLOUD_DOCS_DOMAIN}') через Tavily не дал результатов по запросу: '{query}'."

            # Форматируем результаты в строку
            formatted_response = f"Результаты поиска по документации Yandex Cloud для запроса '{query}':\n\n"
            for i, result in enumerate(raw_results, 1):
                title = result.get("title", "Без заголовка")
                url = result.get("url", "URL не указан")
                content_snippet = result.get("content", "Нет содержимого")

                # Обрезаем слишком длинные сниппеты, если нужно
                max_snippet_length = 500
                if len(content_snippet) > max_snippet_length:
                    content_snippet = content_snippet[:max_snippet_length] + "..."

                formatted_response += f"{i}. {title}\n"
                formatted_response += f"   Источник: {url}\n"
                formatted_response += f"   Фрагмент: {content_snippet}\n\n"

            return formatted_response.strip()

        except Exception as e:
            logger.error(
                f"Ошибка при выполнении поиска Tavily для Yandex Cloud: {e}",
                exc_info=True,
            )
            return f"Произошла ошибка при поиске в документации Yandex Cloud через Tavily: {e}"

    async def _arun(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Асинхронная версия выполнения поиска.
        TavilySearchResults поддерживает асинхронный вызов через `ainvoke`.
        """
        if not self.tavily_search_client:
            return "Ошибка: Клиент Tavily Search API не инициализирован. Проверьте TAVILY_API_KEY."

        logger.info(
            f"Асинхронный запуск TavilyYandexCloudSearchTool с запросом: '{query}' для домена '{YANDEX_CLOUD_DOCS_DOMAIN}'"
        )

        try:
            tool_input_args = {
                "query": query,
                "include_domains": [YANDEX_CLOUD_DOCS_DOMAIN],
            }

            raw_results: List[Dict[str, Any]] = await self.tavily_search_client.ainvoke(
                tool_input_args
            )
            logger.debug(
                f"Получены сырые асинхронные результаты от Tavily: {raw_results}"
            )

            if not raw_results:
                return f"Асинхронный поиск по документации Yandex Cloud ('{YANDEX_CLOUD_DOCS_DOMAIN}') через Tavily не дал результатов по запросу: '{query}'."

            formatted_response = f"Результаты асинхронного поиска по документации Yandex Cloud для запроса '{query}':\n\n"
            for i, result in enumerate(raw_results, 1):
                title = result.get("title", "Без заголовка")
                url = result.get("url", "URL не указан")
                content_snippet = result.get("content", "Нет содержимого")
                max_snippet_length = 500
                if len(content_snippet) > max_snippet_length:
                    content_snippet = content_snippet[:max_snippet_length] + "..."
                formatted_response += f"{i}. {title}\n"
                formatted_response += f"   Источник: {url}\n"
                formatted_response += f"   Фрагмент: {content_snippet}\n\n"

            return formatted_response.strip()

        except Exception as e:
            logger.error(
                f"Ошибка при асинхронном выполнении поиска Tavily для Yandex Cloud: {e}",
                exc_info=True,
            )
            return f"Произошла ошибка при асинхронном поиске в документации Yandex Cloud через Tavily: {e}"


# Пример использования (для локального теста, если нужно)
if __name__ == "__main__":
    load_dotenv()  # Убедитесь, что TAVILY_API_KEY есть в .env

    logging.basicConfig(level=logging.INFO)

    # --- Тест синхронного вызова ---
    tool_sync = TavilyYandexCloudSearchTool()
    if tool_sync.tavily_search_client:
        test_query_sync = "Как подключиться к ВМ в Yandex Cloud?"
        print(
            f"--- Тестирование синхронного вызова с запросом: '{test_query_sync}' ---"
        )
        result_sync = tool_sync._run(query=test_query_sync)
        print(result_sync)
    else:
        print("Синхронный клиент Tavily не инициализирован.")

    # --- Тест асинхронного вызова ---
    async def test_async_tavily():
        tool_async = TavilyYandexCloudSearchTool()
        if tool_async.tavily_search_client:
            test_query_async = "Что такое Yandex Managed Service for Kubernetes?"
            print(
                f"--- Тестирование асинхронного вызова с запросом: '{test_query_async}' ---"
            )
            result_async = await tool_async._arun(query=test_query_async)
            print(result_async)
        else:
            print("Асинхронный клиент Tavily не инициализирован.")

    # Запуск асинхронного теста
    import asyncio

    # Убедимся, что есть .env и он загружен
    from dotenv import load_dotenv

    load_dotenv()

    # if os.getenv("TAVILY_API_KEY"):
    #    asyncio.run(test_async_tavily())
    # else:
    #    print("TAVILY_API_KEY не найден. Асинхронный тест не будет запущен.")

    # Для теста только синхронного вызова, если асинхронный не нужен сразу
    if os.getenv("TAVILY_API_KEY") and tool_sync.tavily_search_client:
        pass  # Синхронный тест уже выполнен выше
    elif not os.getenv("TAVILY_API_KEY"):
        print("TAVILY_API_KEY не найден. Проверьте .env файл.")
