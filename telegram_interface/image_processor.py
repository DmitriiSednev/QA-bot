import asyncio
import logging
import io
from typing import Dict, Any, List, Union

import easyocr
from telegram import Message  # Для type hinting
from telegram.ext import ContextTypes

# Импортируем run_agent_for_user из нового места
from .agent_utils import run_agent_for_user  # <--- ИЗМЕНЕННЫЙ ИМПОРТ

logger = logging.getLogger(__name__)


async def process_image_and_run_agent(
    photo_file_id: str,
    user_id: int,
    chat_id: int,
    text_from_caption: str,
    context: ContextTypes.DEFAULT_TYPE,
    processing_msg: Message,  # Сообщение "Получил фото..."
):
    """Обрабатывает изображение (OCR) и затем вызывает агент."""
    try:
        # --- Ленивая инициализация OCR --- #
        ocr_reader = context.application.bot_data.get("ocr_reader")
        if ocr_reader is None:
            logger.info(
                f"OCR Reader не инициализирован. Попытка инициализации для user {user_id} в чате {chat_id}..."
            )
            try:
                await context.bot.send_chat_action(chat_id=chat_id, action="typing")
                ocr_reader = await asyncio.to_thread(
                    easyocr.Reader, ["ru", "en"], gpu=False  # gpu=False для CPU
                )
                context.application.bot_data["ocr_reader"] = ocr_reader
                logger.info(
                    "OCR Reader (easyocr) успешно инициализирован и сохранен в bot_data."
                )
            except Exception as e_ocr_init:
                logger.error(
                    f"Критическая ошибка при инициализации OCR Reader: {e_ocr_init}.",
                    exc_info=True,
                )
                context.application.bot_data["ocr_reader"] = "error"
                await processing_msg.edit_text(
                    "Ошибка инициализации модуля распознавания текста. Обработка изображений временно недоступна."
                )
                return
        elif ocr_reader == "error":
            logger.warning(
                f"OCR Reader ранее не удалось инициализировать. Обработка фото для user {user_id} невозможна."
            )
            await processing_msg.edit_text(
                "Модуль распознавания текста не загружен. Обработка изображений временно недоступна."
            )
            return
        # --- Конец ленивой инициализации OCR ---

        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        photo_file = await context.bot.get_file(photo_file_id)
        file_bytes_io = io.BytesIO()
        await photo_file.download_to_memory(file_bytes_io)
        image_bytes = file_bytes_io.getvalue()

        loop = asyncio.get_running_loop()
        # Используем ThreadPoolExecutor по умолчанию для run_in_executor
        ocr_result = await loop.run_in_executor(None, ocr_reader.readtext, image_bytes)

        if not ocr_result:
            logger.info("Текст на изображении не распознан.")
            await processing_msg.edit_text(
                "Не удалось распознать текст на этом изображении."
            )
            return

        extracted_text = " ".join([res[1] for res in ocr_result])
        logger.info(
            f"Распознанный текст ({len(extracted_text)} chars): {extracted_text[:200]}..."
        )

        # Обновляем сообщение "Получил фото..." на "Текст распознан..."
        await processing_msg.edit_text(
            "Текст с изображения распознан, обрабатываю запрос..."
        )

        if text_from_caption:
            agent_input_text = f"Пользователь прислал картинку с подписью '{text_from_caption}'. Распознанный текст с картинки: '{extracted_text}'. Ответь на вопрос из подписи, используя текст с картинки как контекст."
        else:
            agent_input_text = f"Пользователь прислал картинку. Распознанный текст с картинки: '{extracted_text}'. Проанализируй этот текст или ответь на вопрос, если он есть в тексте."

        # Вызываем run_agent_for_user, который теперь импортирован
        await run_agent_for_user(user_id, chat_id, agent_input_text, context)

    except Exception as e:
        logger.error(
            f"Ошибка при обработке фото и вызове агента для user {user_id}: {e}",
            exc_info=True,
        )
        try:
            await processing_msg.edit_text(
                "Произошла ошибка при обработке изображения."
            )
        except Exception as e_edit:
            logger.error(f"Ошибка при редактировании сообщения об ошибке OCR: {e_edit}")
            await context.bot.send_message(
                chat_id=chat_id, text="Произошла ошибка при обработке изображения."
            )
