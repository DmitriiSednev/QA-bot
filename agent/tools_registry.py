import logging
from typing import Dict, List, Type, Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel

# --- Импорт классов инструментов ---
from .tools import (
    SearchFAQTool,
    AddFAQTool,
    UpdateFAQTool,
    DeleteFAQTool,
    ContextAnalyzerTool,
    ChatHistorySearchTool,
    # YandexSearchApiTool, # Удаляем старый инструмент
    TavilyYandexCloudSearchTool,  # Добавляем новый инструмент
)

# --- Настройка логгера ---
logger = logging.getLogger(__name__)

# --- Определяем "инструменты" ---
# Описания для LLM-роутера
tool_descriptions: Dict[str, str] = {
    "search_faq": "Искать ответ на вопрос пользователя ВНУТРИ базы знаний FAQ (ПЕРВЫЙ ИСТОЧНИК). Использовать для специфичных знаний проекта.",
    "add_faq": "Добавить новую пару вопрос-ответ в базу знаний FAQ.",
    "update_faq": "Изменить существующую запись в FAQ по её ID.",
    "delete_faq": "Удалить запись из FAQ по её ID.",
    "search_chat_history": "Искать похожие вопросы и ответы в истории чата. Использовать, когда нужно найти ранее заданные похожие вопросы.",
    "context_analyzer": "Анализировать контекст разговора и извлекать ключевые темы и вопросы.",
    # Обновляем описание для нового инструмента поиска по документации
    "tavily_yandexcloud_search": (
        "Используй этот инструмент ПЕРВЫМ для поиска ответов на вопросы, касающиеся **Yandex Cloud**, "
        "его сервисов, API, CLI, устранения неполадок и т.д., "
        "используя поиск **по официальной документации** (`yandex.cloud/ru/docs/`) через Tavily. "
        "Возвращает наиболее релевантные фрагменты из документации."
    ),
}

# Используем классы напрямую для создания экземпляров
tool_classes: List[Type] = [
    SearchFAQTool,
    AddFAQTool,
    UpdateFAQTool,
    DeleteFAQTool,
    ChatHistorySearchTool,
    ContextAnalyzerTool,
    # YandexSearchApiTool, # Удаляем старый
    TavilyYandexCloudSearchTool,  # Добавляем новый
]

# Словарь для хранения инструментов по имени
tool_registry: Dict[str, BaseTool] = {}

# Список инструментов для передачи в LLM и ToolExecutor
tools_list: List[BaseTool] = []


# Функция для регистрации инструмента
def register_tool(tool_instance: BaseTool):
    """Регистрирует инструмент в реестре и добавляет в список."""
    tool_registry[tool_instance.name] = tool_instance
    tools_list.append(tool_instance)
    # Проверяем наличие описания
    if tool_instance.name not in tool_descriptions:
        logger.warning(
            f"Инструмент '{tool_instance.name}' ({tool_instance.__class__.__name__}) инициализирован, но его описания нет в tool_descriptions."
        )


# Создаем экземпляры инструментов (можно передавать параметры конфигурации сюда, если нужно)
# Используем try-except на случай ошибок при инстанцировании
for tool_cls in tool_classes:
    try:
        instance = tool_cls()
        register_tool(instance)
    except Exception as e:
        logger.error(
            f"Ошибка при инстанцировании инструмента {tool_cls.__name__}: {e}",
            exc_info=True,
        )

# Убедимся, что для всех описаний есть инструменты
for name in tool_descriptions:
    if name not in tool_registry:
        logger.warning(
            f"Описание для инструмента '{name}' есть, но сам инструмент не найден в реестре или не удалось его инстанцировать."
        )

logger.info(f"Реестр инструментов инициализирован: {list(tool_registry.keys())}")
