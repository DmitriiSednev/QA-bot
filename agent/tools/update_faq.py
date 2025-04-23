import logging
from typing import Type, Optional

from pydantic import BaseModel, Field, validator
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

from database import crud, connection, models

logger = logging.getLogger(__name__)


class UpdateFAQInput(BaseModel):
    """Схема входных данных для UpdateFAQTool."""

    entry_id: int = Field(description="The ID of the FAQ entry to update.")
    question: Optional[str] = Field(
        None,
        description="The new text for the question. Provide only if you want to change it.",
    )
    answer: Optional[str] = Field(
        None,
        description="The new text for the answer. Provide only if you want to change it.",
    )

    @validator("question", "answer")
    def check_at_least_one_field(cls, v, values):
        """Проверяет, что хотя бы одно поле (вопрос или ответ) предоставлено для обновления."""
        # Этот валидатор вызывается для каждого поля, но нам нужна общая проверка
        # Мы можем проверить это в самом _run, так как validator сложнее для такой логики
        return v

    # Можно добавить валидатор, который будет вызван после всех полей:
    @validator("*", pre=True, always=True)  # Не совсем так работает
    def check_fields_after_creation(cls, v, values):
        # Лучше проверку на наличие хотя бы одного поля делать в _run
        # if 'entry_id' in values and not values.get('question') and not values.get('answer'):
        #     raise ValueError("Хотя бы question или answer должны быть указаны для обновления.")
        return v


class UpdateFAQTool(BaseTool):
    """Инструмент для обновления существующей записи в базе знаний FAQ."""

    name: str = "update_faq"
    description: str = (
        "Updates an existing FAQ entry identified by its ID. "
        "You must provide the entry_id and at least one of the new question or answer texts."
    )
    args_schema: Type[BaseModel] = UpdateFAQInput

    def _run(
        self,
        entry_id: int,
        question: Optional[str] = None,
        answer: Optional[str] = None,
    ) -> str:
        logger.info(
            f"Running UpdateFAQTool for ID={entry_id} with Q:'{question is not None}' A:'{answer is not None}'"
        )

        if question is None and answer is None:
            return "Error: You must provide either a new question or a new answer to update."

        try:
            with connection.get_db_session() as db:
                updated_entry = crud.update_faq_entry(
                    db, entry_id=entry_id, question=question, answer=answer
                )
                if updated_entry:
                    logger.info(f"Successfully updated FAQ entry ID={entry_id}.")
                    return f"Successfully updated FAQ entry with ID {entry_id}."
                else:
                    logger.warning(
                        f"FAQ entry with ID={entry_id} not found for update."
                    )
                    return f"Error: FAQ entry with ID {entry_id} not found."
        except Exception as e:
            logger.error(f"Error updating FAQ entry ID={entry_id}: {e}", exc_info=True)
            return f"An error occurred while updating FAQ entry ID {entry_id}: {e}"

    # async def _arun(self, ...) -> str:
    #     # Асинхронная реализация, если потребуется
    #     pass
