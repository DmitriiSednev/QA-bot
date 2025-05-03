"""Change timestamp columns to DateTime(timezone=True)

Revision ID: 65c9aba432dd
Revises: 9663e105e361
Create Date: 2025-05-02 22:48:51.436469

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '65c9aba432dd'
down_revision: Union[str, None] = '9663e105e361'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Изменение типа для admins.created_at
    try: # Добавляем try-except на случай, если тип уже правильный
        op.alter_column(
            'admins',
            'created_at',
            type_=sa.DateTime(timezone=True),
            existing_type=sa.DateTime(timezone=False), # Указываем предполагаемый текущий тип
            postgresql_using="created_at AT TIME ZONE 'UTC'" # Предполагаем, что старые метки были в UTC
        )
    except Exception as e:
        print(f"Skipping admins.created_at alter: {e}")

    # Изменение типа для chat_history.created_at и updated_at
    for column in ['created_at', 'updated_at']:
        try: # Добавляем try-except
            op.alter_column(
                'chat_history',
                column,
                type_=sa.DateTime(timezone=True),
                existing_type=sa.DateTime(timezone=False), # Указываем предполагаемый текущий тип
                postgresql_using=f"{column} AT TIME ZONE 'UTC'" # Предполагаем, что старые метки были в UTC
            )
        except Exception as e:
            print(f"Skipping chat_history.{column} alter: {e}")

    # Изменение типа для faq_entries.created_at
    try: # Добавляем try-except
        op.alter_column(
            'faq_entries',
            'created_at',
            type_=sa.DateTime(timezone=True),
            existing_type=sa.DateTime(timezone=False), # Указываем предполагаемый текущий тип
            postgresql_using="created_at AT TIME ZONE 'UTC'" # Предполагаем, что старые метки были в UTC
        )
    except Exception as e:
        print(f"Skipping faq_entries.created_at alter: {e}")


def downgrade() -> None:
    # Возврат к исходному типу (TIMESTAMP без временной зоны)
    try: # Добавляем try-except
        op.alter_column(
            'admins',
            'created_at',
            type_=sa.DateTime(timezone=False), # Возвращаем тип без таймзоны
            existing_type=sa.DateTime(timezone=True)
        )
    except Exception as e:
        print(f"Skipping admins.created_at downgrade: {e}")

    for column in ['created_at', 'updated_at']:
        try: # Добавляем try-except
            op.alter_column(
                'chat_history',
                column,
                type_=sa.DateTime(timezone=False), # Возвращаем тип без таймзоны
                existing_type=sa.DateTime(timezone=True)
            )
        except Exception as e:
             print(f"Skipping chat_history.{column} downgrade: {e}")

    try: # Добавляем try-except
        op.alter_column(
            'faq_entries',
            'created_at',
            type_=sa.DateTime(timezone=False), # Возвращаем тип без таймзоны
            existing_type=sa.DateTime(timezone=True)
        )
    except Exception as e:
         print(f"Skipping faq_entries.created_at downgrade: {e}")
