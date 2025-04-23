from sqlalchemy import Column, Integer, String, Text
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

    # Дополнительные поля, которые можно добавить позже:
    # created_at = Column(DateTime(timezone=True), server_default=func.now())
    # updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    # source = Column(String(255))

    def __repr__(self):
        return f"<FAQEntry(id={self.id}, question='{self.question[:50]}...')>"


# Если engine и SessionLocal определены здесь, оставляем их
# Например:
# DATABASE_URL = "postgresql://user:password@host:port/db"
# engine = create_engine(DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Если они в connection.py, этот файл содержит только модели.
