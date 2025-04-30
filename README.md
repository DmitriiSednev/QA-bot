# QA-бот на базе YandexGPT

Telegram-бот для ответов на вопросы с использованием YandexGPT, векторного поиска и базы знаний.

## Возможности

- Ответы на вопросы с использованием YandexGPT
- Векторный поиск по базе знаний FAQ
- Распознавание текста с изображений (OCR)
- Кэширование эмбеддингов для оптимизации
- Параллельная обработка запросов
- Мониторинг и метрики через Prometheus
- Автоматическая очистка устаревших FAQ

## Структура проекта

```
QA-bot/
├── agent/              # Логика агента и инструменты
├── config/             # Конфигурационные файлы
├── core/              # Базовые компоненты
├── database/          # Взаимодействие с БД
├── monitoring/        # Мониторинг и метрики
├── tests/            # Тесты
└── utils/            # Вспомогательные функции
```

## Требования

- Python 3.11+
- PostgreSQL с расширением pgvector
- Docker и Docker Compose

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/yourusername/QA-bot.git
cd QA-bot
```

2. Создайте виртуальное окружение и установите зависимости:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# или
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

3. Создайте файл `.env` на основе `.env.example`:
```bash
cp .env.example .env
# Отредактируйте .env, добавив необходимые ключи
```

4. Запустите через Docker Compose:
```bash
docker-compose up -d
```

## Конфигурация

Основные настройки в `.env`:

```env
TELEGRAM_BOT_TOKEN=your_token
API_KEY=your_yandex_gpt_key
API_BASE=https://llm.api.cloud.yandex.net/...
DATABASE_URL=postgresql://user:pass@host:5432/db
ADMIN_USER_IDS=123456,789012  # Telegram ID админов
```

## Использование

### В личных сообщениях
- Отправьте боту вопрос
- Отправьте изображение с текстом для OCR

### В групповых чатах
- Упомяните бота: `@bot_name вопрос`
- Ответьте на сообщение бота

### Команды админов
- `/admin add user_id` - добавить админа
- `/admin remove user_id` - удалить админа
- `/admin list` - список админов
- `/update_docs` - обновить базу знаний

## Тестирование

```bash
pytest tests/
```

## Мониторинг

Метрики доступны в формате Prometheus по адресу `/metrics`. Основные метрики:
- Количество запросов
- Латентность ответов
- Статистика FAQ
- Ошибки и исключения

## Оптимизация

- Кэширование эмбеддингов
- Параллельная обработка контекста
- Батчинг запросов к API
- Очистка устаревших данных

## Лицензия

MIT

## 🤝 Вклад в проект

1. Форкните репозиторий
2. Создайте ветку для фичи (`git checkout -b feature/amazing`)
3. Закоммитьте изменения (`git commit -m 'Add amazing feature'`)
4. Пусните ветку (`git push origin feature/amazing`)
5. Создайте Pull Request 