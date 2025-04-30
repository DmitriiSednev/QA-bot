# Используем более конкретный и свежий тег базового образа
FROM python:3.11.9-slim-bookworm

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл с зависимостями
COPY requirements.txt requirements.txt

# Устанавливаем зависимости
# --no-cache-dir чтобы не хранить кеш pip, экономим место
# --default-timeout=100 увеличиваем таймаут на случай медленного интернета
# Переформатируем команду RUN для надежности переноса строк
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Копируем весь код проекта в рабочую директорию /app
COPY . .

# Указываем команду для запуска приложения при старте контейнера
CMD ["python", "main.py"] 