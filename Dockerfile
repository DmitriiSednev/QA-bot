# Используем последнюю стабильную версию с фиксированным дайджестом
FROM python:3.11-slim-bookworm@sha256:edaf703dce209d774af3ff768fc92b1e32fbba99789465f5bf49665f699a2d35 as builder

# Устанавливаем необходимые пакеты для сборки и обновляем систему
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Создаем виртуальное окружение
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Копируем только файлы зависимостей
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Финальный этап
FROM python:3.11-slim-bookworm@sha256:edaf703dce209d774af3ff768fc92b1e32fbba99789465f5bf49665f699a2d35

# Устанавливаем обновления безопасности и необходимые пакеты
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Копируем виртуальное окружение из builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Создаем непривилегированного пользователя
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

# Устанавливаем переменные окружения
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Переключаемся на непривилегированного пользователя
USER appuser

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем только необходимые файлы
COPY --chown=appuser:appuser . .

# Используем tini как init процесс
ENTRYPOINT ["/usr/bin/tini", "--"]

# Запускаем бота
CMD ["python", "main.py"] 