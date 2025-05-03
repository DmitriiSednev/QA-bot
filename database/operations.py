from sqlalchemy import func
from .models import FAQEntry
from .connection import get_db_session
import logging
from typing import Optional, List
from database.embeddings import get_embeddings  # Импортируем функцию

logger = logging.getLogger(__name__)

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
        logger.error(f"Ошибка при получении статистики: {str(e)}")
        return None 

def get_embeddings_for_text(text: str) -> Optional[List[float]]:
    try:
        return get_embeddings(text)
    except Exception as e:
        logger.error(f"Ошибка при получении эмбеддингов: {e}")
        return None 