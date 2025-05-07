import pytest
import asyncio
from typing import Dict, Any, Generator, List
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from database import crud_faq, models, connection
from agent.tools import AddFAQTool, SearchFAQTool, UpdateFAQTool, DeleteFAQTool
from agent.graph_builder import setup_agent
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
from database.models import Base, FAQEntry
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from langgraph.checkpoint.sqlite import SqliteSaver
from agent.state import AgentState


# Загружаем переменные окружения для тестов
load_dotenv()


# --- Строка подключения к БД для тестов ---
# Используем SQLite в памяти для скорости и изоляции тестов
# Это избавит от необходимости иметь запущенный PostgreSQL для тестов
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
# Если нужны тесты с PostgreSQL (например, для pgvector), используй:
# SQLALCHEMY_DATABASE_URL = os.getenv(
#     "TEST_DATABASE_URL", "postgresql://test:test@localhost:5432/test_db"
# )


engine = create_engine(SQLALCHEMY_DATABASE_URL)  # Движок создается один раз
# Важно: Для SQLite in-memory каждый connect() создает новую БД.
# Поэтому для тестов нужно использовать одно соединение для create_all и сессий.
# Или создавать таблицы перед каждым тестом.

# Фабрика сессий остается прежней
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")  # Используем scope="function" для SQLite in-memory
def test_db() -> Generator[Session, None, None]:
    """Фикстура для тестовой сессии БД SQLite в памяти."""
    # Создаем таблицы перед каждым тестом для чистой БД
    models.Base.metadata.create_all(bind=engine)
    db_session = TestingSessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()
        # Удаляем таблицы после каждого теста
        models.Base.metadata.drop_all(bind=engine)


# Фикстура для создания тестовых данных FAQ
@pytest.fixture
def sample_faq_entries(test_db: Session) -> List[models.FAQEntry]:
    entries_data = [
        {"question": "python question?", "answer": "python answer!"},
        {"question": "docker question?", "answer": "docker answer!"},
        {"question": "test question?", "answer": "test answer!"},
    ]
    created_entries: List[models.FAQEntry] = []
    for entry_data in entries_data:
        db_entry = crud_faq.add_faq_entry(
            test_db, entry_data["question"], entry_data["answer"]
        )
        if db_entry:
            created_entries.append(db_entry)
    return created_entries


# Тесты CRUD операций
def test_add_faq_entry(test_db: Session):
    entry = crud_faq.add_faq_entry(test_db, "test question?", "test answer!")
    assert entry is not None
    assert entry.question == "test question?"
    assert entry.answer == "test answer!"
    assert entry.id is not None


def test_search_faq_entries(
    test_db: Session, sample_faq_entries: List[models.FAQEntry]
):
    assert len(sample_faq_entries) > 0, "Sample FAQ entries were not created"
    results = crud_faq.search_faq_entries(test_db, "python", limit=1)
    assert len(results) >= 1
    assert any("python" in entry.question.lower() for entry in results)


def test_update_faq_entry(test_db: Session, sample_faq_entries: List[models.FAQEntry]):
    assert len(sample_faq_entries) > 0, "Sample FAQ entries were not created"
    entry_to_update = sample_faq_entries[0]
    updated = crud_faq.update_faq_entry(
        test_db,
        entry_id=entry_to_update.id,
        question="updated question?",
        answer="updated answer!",
    )
    assert updated is not None
    assert updated.question == "updated question?"
    assert updated.answer == "updated answer!"


def test_delete_faq_entry(test_db: Session, sample_faq_entries: List[models.FAQEntry]):
    assert len(sample_faq_entries) > 0, "Sample FAQ entries were not created"
    entry_to_delete = sample_faq_entries[0]
    result = crud_faq.delete_faq_entry(test_db, entry_to_delete.id)
    assert result is True
    assert crud_faq.get_faq_entry_by_id(test_db, entry_to_delete.id) is None


# Тесты инструментов
@pytest.fixture
def mock_db_session(test_db: Session, monkeypatch):
    def mock_get_session_context():
        class MockSessionContext:
            def __enter__(self):
                return test_db

            def __exit__(self, type, value, traceback):
                pass

        return MockSessionContext()

    monkeypatch.setattr(connection, "get_db_session", mock_get_session_context)


def test_add_faq_tool(test_db: Session, mock_db_session):
    tool = AddFAQTool()
    result = tool._run("test tool question?", "test tool answer!")
    assert (
        "успешно добавлена" in result.lower() or "successfully added" in result.lower()
    )
    entries = crud_faq.search_faq_entries(test_db, "test tool question?", limit=1)
    assert len(entries) > 0
    assert entries[0].question == "test tool question?"


def test_search_faq_tool(
    test_db: Session, sample_faq_entries: List[models.FAQEntry], mock_db_session
):
    assert (
        len(sample_faq_entries) > 0
    ), "Sample FAQ entries were not created for search tool test"
    tool = SearchFAQTool()
    query_text = sample_faq_entries[0].question
    expected_answer = sample_faq_entries[0].answer
    result = tool._run(query=query_text)
    assert isinstance(result, str)
    assert query_text in result
    assert expected_answer in result


# Тесты агента
@pytest.mark.asyncio
async def test_agent_flow():
    agent_tuple = await setup_agent()
    assert agent_tuple is not None, "Agent setup failed"
    agent_app, checkpoint_conn = agent_tuple
    assert agent_app is not None, "Agent app is None after setup"
    assert checkpoint_conn is not None, "Checkpoint DB connection is None after setup"

    initial_state: Dict[str, Any] = {
        "messages": [HumanMessage(content="Как использовать Python?")],
        "user_id": 123,
    }
    config = {"configurable": {"thread_id": "pytest_thread_1"}}

    try:
        final_state = await agent_app.ainvoke(initial_state, config=config)
        assert final_state is not None
        assert "messages" in final_state
        assert any(
            isinstance(msg, HumanMessage) for msg in final_state["messages"]
        ), "No HumanMessage in final state"
        if final_state["messages"]:
            assert isinstance(final_state["messages"][-1].content, str)
        else:
            pytest.fail("Agent returned no messages in final_state")

    finally:
        if checkpoint_conn:
            await checkpoint_conn.close()
