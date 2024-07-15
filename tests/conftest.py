import os

import pytest
from pytest_asyncio import fixture
from typedai import TypedAI

# devs will want to use their key when recording new tests, but we want to allow running tests without a key set
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "dummy_key"


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["authorization"],
        "record_mode": "once",
    }


@fixture()
def typed_ai():
    return TypedAI(default_model="gpt-3.5-turbo")
