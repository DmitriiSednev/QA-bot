# QA Bot

Бот для ответов на вопросы с использованием AI.

## Структура проекта

```
.
├── agent/                    # Модуль агента
│   ├── tools/               # Инструменты агента
│   │   ├── __init__.py
│   │   ├── base_search_tool.py    # Базовый класс для инструментов поиска
│   │   ├── search_faq.py          # Поиск по FAQ
│   │   ├── add_faq.py             # Добавление в FAQ
│   │   ├── update_faq.py          # Обновление FAQ
│   │   ├── delete_faq.py          # Удаление из FAQ
│   │   ├── context_analyzer.py    # Анализ контекста
│   │   ├── yandex_cloud_search.py # Поиск в Yandex Cloud
│   │   └── chat_history_search.py # Поиск по истории чата
│   ├── agent_executor.py    # Исполнитель агента
│   └── state.py             # Состояние агента
│
├── database/                # Модуль базы данных
│   ├── __init__.py
│   ├── connection.py       # Подключение к БД
│   ├── crud.py            # CRUD операции
│   └── models.py          # Модели данных
│
├── bot/                    # Модуль бота
│   ├── __init__.py
│   ├── bot.py             # Основной класс бота
│   └── handlers.py        # Обработчики команд
│
├── tests/                  # Тесты
│   ├── __init__.py
│   ├── test_tools.py      # Тесты инструментов
│   └── test_bot.py        # Тесты бота
│
├── .env                    # Переменные окружения
├── .gitignore             # Игнорируемые файлы
├── docker-compose.yml     # Конфигурация Docker
├── Dockerfile             # Сборка Docker
├── requirements.txt       # Зависимости
└── README.md              # Документация
```

## Установка

1. Клонируйте репозиторий
2. Создайте виртуальное окружение: `python -m venv venv`
3. Активируйте окружение: `source venv/bin/activate` (Linux/Mac) или `venv\Scripts\activate` (Windows)
4. Установите зависимости: `pip install -r requirements.txt`
5. Создайте файл `.env` с необходимыми переменными окружения
6. Запустите бота: `python -m bot.bot`

## Docker

Для запуска в Docker:

```bash
docker-compose up -d
```

## Тестирование

```bash
pytest
```

## Лицензия

MIT 