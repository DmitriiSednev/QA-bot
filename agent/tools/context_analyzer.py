import logging
from typing import Type, Dict, Any, Optional, List
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import numpy as np

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

logger = logging.getLogger(__name__)


class ContextAnalyzerInput(BaseModel):
    """Входные данные для анализа контекста."""
    message_text: str = Field(description="Текст сообщения для анализа")
    chat_type: str = Field(description="Тип чата (private, group, supergroup)")
    is_mentioned: bool = Field(description="Упомянут ли бот в сообщении")
    is_reply_to_bot: bool = Field(description="Является ли сообщение ответом на сообщение бота")
    previous_messages: list = Field(description="История последних сообщений в чате")
    last_response_time: datetime | None = Field(description="Время последнего ответа бота")


class ContextAnalyzerTool(BaseTool):
    """Инструмент для анализа контекста сообщения и определения необходимости ответа."""
    
    name: str = "context_analyzer"
    description: str = (
        "Анализирует контекст сообщения и определяет, нужно ли боту отвечать. "
        "Учитывает тип чата, упоминания, историю сообщений и другие факторы."
    )
    args_schema: Type[BaseModel] = ContextAnalyzerInput

    def _analyze_sentiment(self, message_text: str) -> Dict[str, float]:
        """Анализирует эмоциональный тон сообщения."""
        sentiment_keywords = {
            "positive": ["спасибо", "отлично", "хорошо", "помог"],
            "negative": ["плохо", "ошибка", "не работает", "неправильно"]
        }
        
        sentiment_scores = {"positive": 0.0, "negative": 0.0}
        for category, keywords in sentiment_keywords.items():
            for keyword in keywords:
                if keyword in message_text.lower():
                    sentiment_scores[category] += 1.0
        
        return sentiment_scores

    def _analyze_complexity(self, message_text: str) -> float:
        """Анализирует сложность вопроса."""
        complexity_indicators = {
            "сложные_слова": ["конфигурация", "интеграция", "оптимизация"],
            "вопросы": ["как", "почему", "зачем", "что если"],
            "длина": len(message_text.split())
        }
        
        score = 0.0
        for word in complexity_indicators["сложные_слова"]:
            if word in message_text.lower():
                score += 0.3
                
        for question in complexity_indicators["вопросы"]:
            if question in message_text.lower():
                score += 0.2
                
        if complexity_indicators["длина"] > 20:
            score += 0.2
            
        return min(score, 1.0)

    def _analyze_time_context(self, message_time: datetime) -> Dict[str, Any]:
        """Анализирует временной контекст сообщения."""
        now = datetime.now()
        time_diff = now - message_time
        
        return {
            "is_urgent": time_diff.total_seconds() < 300,  # 5 минут
            "is_recent": time_diff.total_seconds() < 3600,  # 1 час
            "time_diff_seconds": time_diff.total_seconds()
        }

    def _run(
        self,
        message_text: str,
        chat_type: str,
        is_mentioned: bool,
        is_reply_to_bot: bool,
        previous_messages: list,
        last_response_time: datetime | None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> Dict[str, Any]:
        """Анализирует контекст и возвращает решение о необходимости ответа."""
        logger.info(f"Анализ контекста для сообщения: {message_text[:100]}...")

        # Базовые правила
        should_respond = False
        confidence = 0.0

        # 1. В личных сообщениях всегда отвечаем
        if chat_type == "private":
            should_respond = True
            confidence = 1.0
            return {"should_respond": should_respond, "confidence": confidence}

        # 2. Если упомянут или ответ на бота - отвечаем
        if is_mentioned or is_reply_to_bot:
            should_respond = True
            confidence = 0.9
            return {"should_respond": should_respond, "confidence": confidence}

        # 3. Анализ текста сообщения
        sentiment = self._analyze_sentiment(message_text)
        complexity = self._analyze_complexity(message_text)
        time_context = self._analyze_time_context(datetime.now())

        # 4. Анализ предыдущих сообщений
        if previous_messages:
            # Проверяем, был ли предыдущий вопрос без ответа
            last_user_message = next(
                (msg for msg in reversed(previous_messages) if msg.get("from_user")),
                None
            )
            if last_user_message and any(
                keyword in last_user_message.get("text", "").lower()
                for keyword in ["как", "что", "почему", "когда", "где", "кто", "зачем"]
            ):
                confidence += 0.2

        # 5. Проверка времени с последнего ответа
        if last_response_time:
            time_since_last_response = datetime.now() - last_response_time
            if time_since_last_response > timedelta(minutes=5):
                confidence += 0.1

        # 6. Проверка на спам/флуд
        if confidence > 0.5:
            should_respond = True

        return {
            "should_respond": should_respond,
            "confidence": confidence,
            "sentiment": sentiment,
            "complexity": complexity,
            "time_context": time_context,
            "reason": "Анализ контекста сообщения"
        }

class ContextCache:
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self._cache: Dict[str, Dict] = {}
        self._max_size = max_size
        self._ttl = ttl

    def get(self, key: str) -> Optional[Dict]:
        if key in self._cache:
            entry = self._cache[key]
            if datetime.now() - entry['timestamp'] < timedelta(seconds=self._ttl):
                return entry['data']
            del self._cache[key]
        return None

    def set(self, key: str, data: Dict):
        if len(self._cache) >= self._max_size:
            # Удаляем самый старый элемент
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        }

    def clear(self):
        self._cache.clear()

context_cache = ContextCache()

@lru_cache(maxsize=1000)
def analyze_message(message: str) -> Dict:
    """Анализ сообщения с кэшированием."""
    # Проверяем кэш
    cached_result = context_cache.get(message)
    if cached_result is not None:
        return cached_result

    # Параллельная обработка различных аспектов сообщения
    with ThreadPoolExecutor(max_workers=3) as executor:
        sentiment_future = executor.submit(_analyze_sentiment, message)
        intent_future = executor.submit(_analyze_intent, message)
        entities_future = executor.submit(_extract_entities, message)

        result = {
            'sentiment': sentiment_future.result(),
            'intent': intent_future.result(),
            'entities': entities_future.result(),
            'timestamp': datetime.now()
        }

    # Сохраняем в кэш
    context_cache.set(message, result)
    return result

def _analyze_sentiment(message: str) -> float:
    """Анализ тональности сообщения."""
    # Реализация анализа тональности
    return 0.0

def _analyze_intent(message: str) -> str:
    """Анализ намерения в сообщении."""
    # Реализация анализа намерения
    return "unknown"

def _extract_entities(message: str) -> List[str]:
    """Извлечение сущностей из сообщения."""
    # Реализация извлечения сущностей
    return []

def clear_context_cache():
    """Очистка кэша контекста."""
    context_cache.clear() 