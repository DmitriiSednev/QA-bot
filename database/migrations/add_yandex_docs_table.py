"""Добавление таблицы для документации Yandex Cloud

Revision ID: add_yandex_docs_table
Revises: initial
Create Date: 2024-03-14
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = 'add_yandex_docs_table'
down_revision: Union[str, None] = 'initial'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # Создаем таблицу для документации
    op.create_table(
        'yandex_docs_entries',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('path', sa.String(length=500), nullable=False),
        sa.Column('title', sa.String(length=500), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('embedding', Vector(1536), nullable=True),
        sa.Column('last_updated', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Создаем индексы
    op.create_index(op.f('ix_yandex_docs_entries_path'), 'yandex_docs_entries', ['path'], unique=True)
    op.create_index(op.f('ix_yandex_docs_entries_title'), 'yandex_docs_entries', ['title'])
    
    # Создаем индекс для векторного поиска
    op.execute(
        'CREATE INDEX yandex_docs_entries_embedding_idx ON yandex_docs_entries USING ivfflat (embedding vector_cosine_ops)'
    )

def downgrade() -> None:
    # Удаляем индексы
    op.drop_index('yandex_docs_entries_embedding_idx', table_name='yandex_docs_entries')
    op.drop_index(op.f('ix_yandex_docs_entries_title'), table_name='yandex_docs_entries')
    op.drop_index(op.f('ix_yandex_docs_entries_path'), table_name='yandex_docs_entries')
    
    # Удаляем таблицу
    op.drop_table('yandex_docs_entries') 