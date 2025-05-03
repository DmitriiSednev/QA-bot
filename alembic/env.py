import os
from logging.config import fileConfig
# Удалим load_dotenv отсюда, т.к. он есть в connection
# from dotenv import load_dotenv

from sqlalchemy import engine_from_config
from sqlalchemy import pool, text

from alembic import context

# Импортируем модели для таблиц из нашего проекта
from database.models import Base
# from database.models import FAQEntry, Admin, ChatHistory # Можно убрать, Base достаточно

# !!! Импортируем собранный URL из connection !!!
# Это также вызовет load_dotenv() внутри connection.py
from database.connection import DATABASE_URL, engine as connection_engine

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# !!! Устанавливаем URL напрямую из импортированной переменной !!!
config.set_main_option("sqlalchemy.url", DATABASE_URL)

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.

# --- Функция для фильтрации объектов при автогенерации ---
def include_object(object, name, type_, reflected, compare_to):
    """
    Should you include this object?
    """
    # Игнорируем таблицы и объекты из системных схем Supabase/PostgreSQL
    if type_ == "table" and object.schema not in [None, 'public']:
        return False
    # Игнорируем служебную таблицу Alembic
    if type_ == "table" and name == "alembic_version" and object.schema == 'public':
        return False

    # Пропускаем все остальное в схеме public (или без схемы)
    return True


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    # url = config.get_main_option("sqlalchemy.url") # Уже установлено выше
    context.configure(
        url=DATABASE_URL, # Используем импортированный URL
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
        compare_type=True
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    # Создаем engine, используя URL, установленный ранее
    # Используем существующий engine из connection, чтобы избежать повторного создания
    connectable = connection_engine

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
            compare_type=True,
            transaction_per_migration=True
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
