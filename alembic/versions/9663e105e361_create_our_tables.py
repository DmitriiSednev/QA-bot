"""Create our tables

Revision ID: 9663e105e361
Revises: 
Create Date: 2025-05-02 21:09:23.852586

"""
from typing import Sequence, Union

from alembic import op, context
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# Добавляем импорт pgvector
try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    # Если pgvector не установлен, создаем заглушку для миграций
    class Vector:
        def __init__(self, dim):
            self.dim = dim

# revision identifiers, used by Alembic.
revision: str = '9663e105e361'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Создаем только наши таблицы, не трогая системные таблицы Supabase
    
    # 1. Таблица администраторов (в своей транзакции)
    try:
        with context.begin_transaction():
            op.create_table('admins',
                sa.Column('id', sa.Integer(), nullable=False),
                sa.Column('user_id', sa.Integer(), nullable=False),
                sa.Column('username', sa.String(length=255), nullable=True),
                sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
                sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
                sa.PrimaryKeyConstraint('id')
            )
            op.create_index(op.f('ix_admins_id'), 'admins', ['id'], unique=False)
            op.create_index(op.f('ix_admins_user_id'), 'admins', ['user_id'], unique=True)
    except Exception as e:
        print(f"Skipping admins table creation (already exists or error): {e}")
    
    # 2. Таблица истории чата (в своей транзакции)
    try:
        with context.begin_transaction():
            op.create_table('chat_history',
                sa.Column('id', sa.Integer(), nullable=False),
                sa.Column('chat_id', sa.Integer(), nullable=False),
                sa.Column('user_id', sa.Integer(), nullable=False),
                sa.Column('message_text', sa.Text(), nullable=False),
                sa.Column('response_text', sa.Text(), nullable=True),
                sa.Column('embedding', Vector(dim=1536), nullable=True),
                sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
                sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
                sa.Column('is_answered', sa.Boolean(), server_default='false', nullable=True),
                sa.Column('answer_quality', sa.Integer(), nullable=True),
                sa.PrimaryKeyConstraint('id')
            )
            op.create_index(op.f('ix_chat_history_chat_id'), 'chat_history', ['chat_id'], unique=False)
            op.create_index(op.f('ix_chat_history_id'), 'chat_history', ['id'], unique=False)
            op.create_index(op.f('ix_chat_history_user_id'), 'chat_history', ['user_id'], unique=False)
    except Exception as e:
        print(f"Skipping chat_history table creation (already exists or error): {e}")

    # 3. Таблица FAQ_TEST (в своей транзакции)
    test_table_name = 'faq_entries' # Новое имя
    try:
        with context.begin_transaction():
            op.create_table(test_table_name,
                sa.Column('id', sa.Integer(), nullable=False),
                sa.Column('question', sa.Text(), nullable=False),
                sa.Column('answer', sa.Text(), nullable=False),
                sa.Column('embedding', Vector(dim=1536), nullable=True),
                sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
                sa.PrimaryKeyConstraint('id')
            )
            op.create_index(op.f(f'ix_{test_table_name}_id'), test_table_name, ['id'], unique=False)
            op.create_index(op.f(f'ix_{test_table_name}_question'), test_table_name, ['question'], unique=False)
            print(f"Successfully created table {test_table_name}")
    except Exception as e:
        print(f"Skipping {test_table_name} table creation (already exists or error): {e}")


def downgrade() -> None:
    """Downgrade schema."""
    test_table_name = 'faq_entries_test' # Новое имя
    # Удаляем только наши таблицы в обратном порядке
    try:
        with context.begin_transaction():
            op.drop_index(op.f(f'ix_{test_table_name}_question'), table_name=test_table_name)
            op.drop_index(op.f(f'ix_{test_table_name}_id'), table_name=test_table_name)
            op.drop_table(test_table_name)
    except Exception as e:
        print(f"Skipping drop of {test_table_name} (doesn't exist or error): {e}")

    try:
        with context.begin_transaction():
            op.drop_index(op.f('ix_chat_history_user_id'), table_name='chat_history')
            op.drop_index(op.f('ix_chat_history_id'), table_name='chat_history')
            op.drop_index(op.f('ix_chat_history_chat_id'), table_name='chat_history')
            op.drop_table('chat_history')
    except Exception as e:
        print(f"Skipping drop of chat_history (doesn't exist or error): {e}")

    try:
        with context.begin_transaction():
            op.drop_index(op.f('ix_admins_user_id'), table_name='admins')
            op.drop_index(op.f('ix_admins_id'), table_name='admins')
            op.drop_table('admins')
    except Exception as e:
        print(f"Skipping drop of admins (doesn't exist or error): {e}")
