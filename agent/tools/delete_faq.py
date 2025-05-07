import logging
from typing import Type

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from database import connection
from database.crud_faq import delete_faq_entry

logger = logging.getLogger(__name__)


class DeleteFAQInput(BaseModel):
    entry_id: int = Field(description="The ID of the FAQ entry to delete.")


class DeleteFAQTool(BaseTool):
    name: str = "delete_faq"
    description: str = "Deletes an existing FAQ entry identified by its ID."
    args_schema: Type[BaseModel] = DeleteFAQInput

    def _run(self, entry_id: int) -> str:
        logger.info(f"Running DeleteFAQTool for ID={entry_id}")

        try:
            with connection.get_db_session() as db:
                if not db:
                    return "Ошибка: Не удалось получить сессию базы данных."
                success = delete_faq_entry(db, entry_id=entry_id)
                if success:
                    logger.info(f"Successfully deleted FAQ entry ID={entry_id}.")
                    return f"Successfully deleted FAQ entry with ID {entry_id}."
                else:
                    # crud.delete_faq_entry возвращает False, если запись не найдена или ошибка
                    logger.warning(
                        f"FAQ entry with ID={entry_id} not found or error during deletion."
                    )
                    return (
                        f"Error: Could not delete FAQ entry with ID {entry_id}. "
                        f"It might not exist or an error occurred."
                    )
        except Exception as e:
            logger.error(f"Error deleting FAQ entry ID={entry_id}: {e}", exc_info=True)
            return f"An error occurred while deleting FAQ entry ID {entry_id}: {e}"

    # async def _arun(self, ...) -> str:
    #     # Асинхронная реализация, если потребуется
    #     pass
