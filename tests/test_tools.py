import pytest
import asyncio
from typing import Dict, Any, Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from database import crud_faq, models, connection
from agent.tools import AddFAQTool, SearchFAQTool, UpdateFAQTool, DeleteFAQTool
from agent.agent_executor import setup_agent
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

# Загружаем переменные окружения для тестов
load_dotenv()


# Фикстура для тестовой БД
@pytest.fixture(scope="session")
def test_engine():
    SQLALCHEMY_DATABASE_URL = os.getenv(
        "TEST_DATABASE_URL", "postgresql://test:test@localhost:5432/test_db"
    )
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    try:
        # Создаем все таблицы
        models.Base.metadata.create_all(bind=engine)
        yield engine
    finally:
        # Удаляем все таблицы после тестов
        models.Base.metadata.drop_all(bind=engine)


@pytest.fixture
def test_db(test_engine) -> Generator[Session, None, None]:
    """Фикстура для тестовой сессии БД."""
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=test_engine
    )
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.rollback()
        db.close()


# Фикстура для создания тестовых данных
@pytest.fixture
def sample_faq_entries(test_db: Session):
    entries = [
        {"question": "python question?", "answer": "python answer!"},
        {"question": "docker question?", "answer": "docker answer!"},
        {"question": "test question?", "answer": "test answer!"},
    ]
    created_entries = []
    for entry in entries:
        db_entry = crud_faq.add_faq_entry(test_db, entry["question"], entry["answer"])
        created_entries.append(db_entry)
    return created_entries


# Тесты CRUD операций
def test_add_faq_entry(test_db: Session):
    entry = crud_faq.add_faq_entry(test_db, "test question?", "test answer!")
    assert entry is not None
    assert entry.question == "test question?"
    assert entry.answer == "test answer!"
    assert entry.id is not None


def test_search_faq_entries(test_db: Session, sample_faq_entries):
    results = crud_faq.search_faq_entries(test_db, "python")
    assert len(results) > 0
    assert any("python" in entry.question.lower() for entry in results)


def test_update_faq_entry(test_db: Session, sample_faq_entries):
    entry = sample_faq_entries[0]
    updated = crud_faq.update_faq_entry(
        test_db,
        entry_id=entry.id,
        question="updated question?",
        answer="updated answer!",
    )
    assert updated is not None
    assert updated.question == "updated question?"
    assert updated.answer == "updated answer!"


def test_delete_faq_entry(test_db: Session, sample_faq_entries):
    entry = sample_faq_entries[0]
    result = crud_faq.delete_faq_entry(test_db, entry.id)
    assert result is True
    assert crud_faq.get_faq_entry_by_id(test_db, entry.id) is None


# Тесты инструментов
def test_add_faq_tool(test_db: Session, monkeypatch):
    def mock_get_db_session():
        return test_db

    monkeypatch.setattr(connection, "get_db_session", mock_get_db_session)

    tool = AddFAQTool()
    result = tool._run("test tool question?", "test tool answer!")
    assert "успешно" in result.lower()

    # Проверяем, что запись действительно добавлена
    entries = crud_faq.search_faq_entries(test_db, "test tool question?")
    assert len(entries) > 0
    assert entries[0].question == "test tool question?"


def test_search_faq_tool(test_db: Session, sample_faq_entries, monkeypatch):
    def mock_get_db_session():
        return test_db

    monkeypatch.setattr(connection, "get_db_session", mock_get_db_session)

    tool = SearchFAQTool()
    result = tool._run("python")
    assert isinstance(result, str)
    assert "python question?" in result
    assert "python answer!" in result


# Тесты агента
@pytest.mark.asyncio
async def test_agent_flow():
    agent = setup_agent()
    assert agent is not None

    state: Dict[str, Any] = {
        "messages": [HumanMessage(content="Как использовать Python?")],
        "user_id": 123,
    }

    result = await agent.ainvoke(state)
    assert "messages" in result
    assert len(result["messages"]) > 0
    assert isinstance(result["messages"][-1].content, str)
