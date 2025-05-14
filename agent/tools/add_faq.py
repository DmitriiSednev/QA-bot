import logging
from typing import Type

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

from database import connection
from database.crud_faq import add_faq_entry
from database.embeddings import get_embeddings

logger = logging.getLogger(__name__)


class AddFAQInput(BaseModel):
    """Схема входных данных для AddFAQTool."""

    question: str = Field(description="Новый вопрос для добавления в FAQ.")
    answer: str = Field(description="Ответ на новый вопрос.")


class AddFAQTool(BaseTool):
    """Инструмент для добавления новой пары вопрос-ответ в базу знаний FAQ."""

    name: str = "add_faq"
    description: str = (
        "Используй этот инструмент, когда нужно добавить новую информацию "
        "в базу знаний FAQ. Например, если пользователь явно просит сохранить "
        "вопрос и ответ, или если агент сам решил, что найденный ответ стоит сохранить."
        " Требует точный текст вопроса и ответа."
    )
    args_schema: Type[BaseModel] = AddFAQInput

    def _run(
        self,
        question: str,
        answer: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Добавляет запись в FAQ."""
        logger.info(f"Запуск AddFAQTool с вопросом: '{question}'")

        try:
            with connection.get_db_session() as db:
                if not db:
                    return "Ошибка: Не удалось получить сессию базы данных."

                db_entry = add_faq_entry(
                    db, question=question, answer=answer, embedding=None
                )

                if db_entry:
                    logger.info(
                        f"Запись FAQ (ID: {db_entry.id}) создана, генерирую эмбеддинг..."
                    )
                    embedding_vector = get_embeddings(db_entry.question)
                    if embedding_vector and embedding_vector[0]:
                        db_entry.embedding = embedding_vector[0]
                        try:
                            db.commit()
                            db.refresh(db_entry)
                            logger.info(
                                f"Эмбеддинг для FAQ (ID: {db_entry.id}) успешно сохранен."
                            )
                            return f"Запись FAQ (ID: {db_entry.id}) с эмбеддингом успешно добавлена в базу знаний."
                        except Exception as e_commit:
                            logger.error(
                                f"Ошибка при сохранении эмбеддинга для FAQ ID {db_entry.id}: {e_commit}",
                                exc_info=True,
                            )
                            return f"Запись FAQ (ID: {db_entry.id}) добавлена, но не удалось сохранить эмбеддинг: {e_commit}"
                    else:
                        logger.warning(
                            f"Не удалось сгенерировать эмбеддинг для вопроса FAQ ID {db_entry.id}: '{db_entry.question}'. Запись добавлена без эмбеддинга."
                        )
                        return f"Запись FAQ (ID: {db_entry.id}) успешно добавлена, но не удалось сгенерировать эмбеддинг."
                else:
                    return "Не удалось добавить запись в FAQ из-за внутренней ошибки."

        except Exception as e:
            logger.error(
                f"Ошибка в AddFAQTool при добавлении вопроса '{question}': {e}",
                exc_info=True,
            )
            return f"Произошла ошибка при добавлении записи в FAQ: {e}"
