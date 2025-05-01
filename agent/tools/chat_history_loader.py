import logging
from typing import Type, Dict, Any, Optional, List
from datetime import datetime, timedelta
from telethon import TelegramClient
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
import os
from dotenv import load_dotenv

from database import crud, connection

logger = logging.getLogger(__name__)
load_dotenv()

class ChatHistoryLoaderInput(BaseModel):
    """Схема входных данных для ChatHistoryLoaderTool."""
    chat_id: int = Field(description="ID чата для загрузки истории")
    days_to_load: int = Field(default=30, description="Количество дней истории для загрузки")

class ChatHistoryLoaderTool(BaseTool):
    """Инструмент для загрузки истории чата при добавлении бота в группу."""

    name: str = "load_chat_history"
    description: str = (
        "Используй этот инструмент для загрузки истории чата при добавлении бота в группу. "
        "Это позволит боту сразу иметь доступ к предыдущим сообщениям в чате."
    )
    args_schema: Type[BaseModel] = ChatHistoryLoaderInput

    def __init__(self):
        super().__init__()
        self.client = None
        self._init_client()

    def _init_client(self):
        """Инициализация клиента Telegram."""
        try:
            api_id = int(os.getenv('TELEGRAM_APP_ID', ''))
            api_hash = os.getenv('TELEGRAM_APP_HASH', '')
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
            
            if not all([api_id, api_hash, bot_token]):
                raise ValueError("Не все необходимые переменные окружения установлены")
            
            self.client = TelegramClient(
                'qa_bot_session',
                api_id,
                api_hash
            )
            self.client.start(bot_token=bot_token)
            logger.info("Telethon клиент успешно инициализирован")
            
        except Exception as e:
            logger.error(f"Ошибка при инициализации Telethon клиента: {e}")
            raise

    def _run(
        self,
        chat_id: int,
        days_to_load: int = 30,
        run_manager: CallbackManagerForToolRun | None = None
    ) -> str:
        """Загружает историю чата."""
        try:
            if not self.client:
                self._init_client()

            # Определяем временное окно
            start_date = datetime.now() - timedelta(days=days_to_load)
            
            # Получаем историю сообщений
            messages = []
            for message in self.client.iter_messages(
                chat_id,
                min_id=0,
                reverse=True,
                offset_date=start_date
            ):
                if message.text:  # Пропускаем сообщения без текста
                    messages.append({
                        'chat_id': chat_id,
                        'user_id': message.sender_id,
                        'message_text': message.text,
                        'created_at': message.date
                    })

            if not messages:
                return f"В чате {chat_id} не найдено сообщений за последние {days_to_load} дней."

            # Сохраняем сообщения в базу данных
            with connection.get_db_session() as db:
                crud.add_chat_history_batch(db, messages)
            
            return f"Успешно загружено {len(messages)} сообщений из чата {chat_id} за последние {days_to_load} дней."

        except Exception as e:
            logger.error(f"Ошибка при загрузке истории чата: {e}", exc_info=True)
            return f"Произошла ошибка при загрузке истории чата: {e}" 