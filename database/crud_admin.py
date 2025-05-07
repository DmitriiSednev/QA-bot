import logging
from typing import List, Optional

from sqlalchemy.orm import Session
from sqlalchemy import select, update as sql_update
from sqlalchemy.exc import SQLAlchemyError

from . import models

logger = logging.getLogger(__name__)

# Функции, перенесенные из crud.py, относящиеся к Admin

def get_admin_by_user_id(db: Session, user_id: int) -> Optional[models.Admin]:
    """Получает админа по его Telegram user_id."""
    try:
        stmt = select(models.Admin).where(models.Admin.user_id == user_id)
        return db.execute(stmt).scalar_one_or_none()
    except SQLAlchemyError as e:
        logger.error(f"Ошибка SQLAlchemy при получении админа по user_id={user_id}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Неожиданная ошибка при получении админа user_id={user_id}: {e}", exc_info=True)
        return None

def get_admin_by_username(db: Session, username: str) -> Optional[models.Admin]:
    """Получает админа по его Telegram username."""
    if not username:
        return None
    try:
        stmt = select(models.Admin).where(models.Admin.username == username)
        return db.execute(stmt).scalar_one_or_none()
    except SQLAlchemyError as e:
        logger.error(f"Ошибка SQLAlchemy при получении админа по username='{username}': {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Неожиданная ошибка при получении админа username='{username}': {e}", exc_info=True)
        return None

def add_admin(db: Session, user_id: int, username: Optional[str] = None) -> Optional[models.Admin]:
    """Добавляет нового админа или активирует существующего неактивного."""
    try:
        existing_admin = get_admin_by_user_id(db, user_id)
        if existing_admin:
            if not existing_admin.is_active:
                logger.warning(f"Админ user_id={user_id} уже существует, но неактивен. Активируем и обновляем username...")
                existing_admin.is_active = True
                existing_admin.username = username
                db.commit()
                db.refresh(existing_admin)
                logger.info(f"Админ user_id={user_id} активирован.")
                return existing_admin
            else:
                # Обновляем username, если он изменился у существующего активного админа
                if existing_admin.username != username:
                    logger.info(f"Обновление username для существующего активного админа user_id={user_id}.")
                    existing_admin.username = username
                    db.commit()
                    db.refresh(existing_admin)
                else:
                    logger.info(f"Админ user_id={user_id} уже существует и активен.")
                return existing_admin

        logger.info(f"Создание нового админа: user_id={user_id}, username={username}")
        admin = models.Admin(user_id=user_id, username=username, is_active=True)
        db.add(admin)
        db.commit()
        db.refresh(admin)
        logger.info(f"Новый админ user_id={user_id} успешно добавлен.")
        return admin
    except SQLAlchemyError as e:
        logger.error(f"Ошибка SQLAlchemy при добавлении/активации админа user_id={user_id}: {e}", exc_info=True)
        db.rollback()
        return None
    except Exception as e:
        logger.error(f"Неожиданная ошибка при добавлении/активации админа user_id={user_id}: {e}", exc_info=True)
        db.rollback()
        return None

def get_all_active_admins(db: Session) -> List[models.Admin]:
    """Получает список всех активных админов."""
    try:
        stmt = select(models.Admin).where(models.Admin.is_active == True).order_by(models.Admin.id)
        return db.execute(stmt).scalars().all()
    except SQLAlchemyError as e:
        logger.error(f"Ошибка SQLAlchemy при получении списка админов: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Неожиданная ошибка при получении списка админов: {e}", exc_info=True)
        return []

def deactivate_admin(db: Session, user_id: int) -> bool:
    """Деактивирует админа по user_id."""
    try:
        # Сначала найдем админа
        admin_to_deactivate = db.execute(
            select(models.Admin).where(models.Admin.user_id == user_id)
        ).scalar_one_or_none()

        if not admin_to_deactivate:
            logger.warning(f"Админ user_id={user_id} не найден для деактивации.")
            return False

        if not admin_to_deactivate.is_active:
            logger.warning(f"Админ user_id={user_id} уже неактивен.")
            return True # Считаем успешным, т.к. цель достигнута

        # Деактивируем
        admin_to_deactivate.is_active = False
        db.commit()
        logger.info(f"Админ user_id={user_id} успешно деактивирован.")
        return True

    except SQLAlchemyError as e:
        logger.error(f"Ошибка SQLAlchemy при деактивации админа user_id={user_id}: {e}", exc_info=True)
        db.rollback()
        return False
    except Exception as e:
        logger.error(f"Неожиданная ошибка при деактивации админа user_id={user_id}: {e}", exc_info=True)
        db.rollback()
        return False 