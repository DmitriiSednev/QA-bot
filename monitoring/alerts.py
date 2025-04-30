from prometheus_client import Counter
import logging
from typing import Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Счетчики для алертов
ALERTS = Counter('qa_bot_alerts_total', 'Total alerts', ['type', 'severity'])

class AlertManager:
    def __init__(self):
        self.last_alert_time: Dict[str, datetime] = {}
        self.alert_cooldown = timedelta(minutes=5)  # Защита от флуда алертов

    def check_alert_cooldown(self, alert_type: str) -> bool:
        """Проверяет, можно ли отправлять алерт."""
        last_time = self.last_alert_time.get(alert_type)
        if not last_time or datetime.now() - last_time > self.alert_cooldown:
            self.last_alert_time[alert_type] = datetime.now()
            return True
        return False

    def send_alert(self, alert_type: str, severity: str, message: str, data: Dict[str, Any] = None):
        """Отправляет алерт."""
        if not self.check_alert_cooldown(alert_type):
            logger.debug(f"Alert {alert_type} skipped due to cooldown")
            return

        ALERTS.labels(type=alert_type, severity=severity).inc()
        logger.warning(f"ALERT [{severity}] {alert_type}: {message}")
        if data:
            logger.warning(f"Alert data: {data}")

    def check_error_rate(self, error_count: int, threshold: int = 10):
        """Проверяет количество ошибок."""
        if error_count > threshold:
            self.send_alert(
                "error_rate",
                "critical",
                f"High error rate detected: {error_count} errors",
                {"error_count": error_count}
            )

    def check_response_time(self, latency: float, threshold: float = 5.0):
        """Проверяет время ответа."""
        if latency > threshold:
            self.send_alert(
                "response_time",
                "warning",
                f"High response time: {latency:.2f}s",
                {"latency": latency}
            )

    def check_faq_health(self, entries_count: int, min_entries: int = 10):
        """Проверяет здоровье FAQ."""
        if entries_count < min_entries:
            self.send_alert(
                "faq_health",
                "warning",
                f"Low FAQ entries count: {entries_count}",
                {"entries_count": entries_count}
            )

# Глобальный экземпляр менеджера алертов
alert_manager = AlertManager() 