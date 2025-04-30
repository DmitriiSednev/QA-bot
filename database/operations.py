from sqlalchemy import func
from .models import FAQEntry
from .connection import get_db_session
import logging

def get_knowledge_base_stats():
    """Получение статистики по базе знаний"""
    try:
        with get_db_session() as session:
            total_entries = session.query(func.count(FAQEntry.id)).scalar()
            latest_entry = session.query(FAQEntry).order_by(FAQEntry.created_at.desc()).first()
            
            return {
                "total_entries": total_entries,
                "latest_entry_date": latest_entry.created_at if latest_entry else None
            }
    except Exception as e:
        logging.error(f"Ошибка при получении статистики: {str(e)}")
        return None 