from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import Dict, Any
import logging
from database import connection, operations

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Проверка здоровья сервиса."""
    try:
        # Проверяем подключение к БД
        db_ok = connection.check_db_connection()
        if not db_ok:
            raise HTTPException(status_code=503, detail="Database connection failed")

        # Получаем статистику
        stats = operations.get_knowledge_base_stats()
        if not stats:
            raise HTTPException(status_code=503, detail="Failed to get database stats")

        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "database": {
                "status": "ok",
                "total_entries": stats["total_entries"],
                "latest_entry": stats["latest_entry_date"].isoformat() if stats["latest_entry_date"] else None
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=str(e))

@router.get("/metrics")
async def metrics() -> Dict[str, Any]:
    """Возвращает метрики в формате Prometheus."""
    try:
        from prometheus_client import generate_latest
        from fastapi.responses import Response
        return Response(generate_latest(), media_type="text/plain")
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 