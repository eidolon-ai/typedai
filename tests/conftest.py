import pytest
from pytest_asyncio import fixture
from typedai import TypedAI


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["authorization"],
        "record_mode": "once",
    }


@fixture()
def typed_ai():
    return TypedAI(default_model="gpt-3.5-turbo")
