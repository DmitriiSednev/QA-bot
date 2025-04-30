from prometheus_client import Counter, Histogram, Gauge
import logging

logger = logging.getLogger(__name__)

# Метрики запросов
REQUESTS = Counter('qa_bot_requests_total', 'Total requests', ['status'])
REQUEST_LATENCY = Histogram('qa_bot_request_latency_seconds', 'Request latency')

# Метрики БД
DB_OPERATIONS = Counter('qa_bot_db_operations_total', 'Total DB operations', ['operation'])
DB_LATENCY = Histogram('qa_bot_db_latency_seconds', 'DB operation latency')

# Метрики FAQ
FAQ_ENTRIES = Gauge('qa_bot_faq_entries_total', 'Total FAQ entries')
FAQ_SEARCHES = Counter('qa_bot_faq_searches_total', 'Total FAQ searches')
FAQ_UPDATES = Counter('qa_bot_faq_updates_total', 'Total FAQ updates')

# Метрики ошибок
ERRORS = Counter('qa_bot_errors_total', 'Total errors', ['type'])

# Метрики агента
AGENT_TOOL_CALLS = Counter('qa_bot_agent_tool_calls_total', 'Total agent tool calls', ['tool'])
AGENT_CONFIDENCE = Histogram('qa_bot_agent_confidence', 'Agent response confidence')

def update_faq_metrics(entries_count: int):
    """Обновляет метрики FAQ."""
    FAQ_ENTRIES.set(entries_count)
    logger.debug(f"Updated FAQ metrics: {entries_count} entries")

def record_error(error_type: str):
    """Записывает ошибку в метрики."""
    ERRORS.labels(type=error_type).inc()
    logger.error(f"Recorded error of type: {error_type}")

def record_tool_call(tool_name: str):
    """Записывает вызов инструмента."""
    AGENT_TOOL_CALLS.labels(tool=tool_name).inc()
    logger.debug(f"Recorded tool call: {tool_name}")

def record_confidence(confidence: float):
    """Записывает уверенность агента."""
    AGENT_CONFIDENCE.observe(confidence)
    logger.debug(f"Recorded agent confidence: {confidence}")
