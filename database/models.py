from sqlalchemy import Column, Integer, String, Text, DateTime, func, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector

# Создаем базовый класс для декларативных моделей
Base = declarative_base()


class FAQEntry(Base):
    """Модель SQLAlchemy для таблицы записей FAQ."""

    __tablename__ = "faq_entries"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False, index=True)  # Индекс по вопросу для поиска
    answer = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)

    # Дополнительные поля, которые можно добавить позже:
    # updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    # source = Column(String(255))

    def __repr__(self):
        return f"<FAQEntry(id={self.id}, question='{self.question[:50]}...')>"


class Admin(Base):
    """Модель SQLAlchemy для таблицы администраторов."""

    __tablename__ = "admins"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, unique=True, nullable=False, index=True)  # Telegram user_id
    username = Column(String(255), nullable=True)  # Telegram username
    created_at = Column(DateTime, default=func.now(), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    def __repr__(self):
        return f"<Admin(id={self.id}, user_id={self.user_id}, username='{self.username}')>"


class ChatHistory(Base):
    """Модель для хранения истории чата с векторными эмбеддингами."""
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, nullable=False, index=True)  # ID чата Telegram
    user_id = Column(Integer, nullable=False, index=True)  # ID пользователя
    message_text = Column(Text, nullable=False)
    response_text = Column(Text, nullable=True)  # Ответ бота
    embedding = Column(Vector(1536), nullable=True)  # Векторное представление сообщения
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    is_answered = Column(Boolean, default=False)  # Был ли дан ответ на вопрос
    answer_quality = Column(Integer, nullable=True)  # Оценка качества ответа

    def __repr__(self):
        return f"<ChatHistory(id={self.id}, chat_id={self.chat_id})>"


# Если engine и SessionLocal определены здесь, оставляем их
# Например:
# DATABASE_URL = "postgresql://user:password@host:port/db"
# engine = create_engine(DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Если они в connection.py, этот файл содержит только модели.
