import json

from openai import BaseModel, OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam
from pytest_asyncio import fixture
from typedai import TypedAI
from typedai.messages import System, User


@fixture()
def typed_ai():
    return TypedAI()


def test_instantiation(typed_ai):
    assert typed_ai


def test_string_completions(typed_ai):
    completion = typed_ai.completions.create(
        model="gpt-3.5-turbo",
        messages=[System(content="You are a helpful assistant"), User(content="What is the capital of France?")],
    )
    assert "paris" in completion.choices[0].message.content.lower()


def test_typed_completions(typed_ai):
    class MyResponseObject(BaseModel):
        philosopher: str
        meaning: str
        bullshit_level: float = .5

    completion = typed_ai.completions.create(
        model="gpt-3.5-turbo",
        messages=[System(content="You are a helpful assistant"), User(content="What is the meaning of life according to Douglas Adams?")],
        response_format=MyResponseObject,
    )
    my_response_object = completion.choices[0].message.content
    assert my_response_object.philosopher == "Douglas Adams"
    assert my_response_object.meaning == "42"


def test_prompt_engineering():
    args = {
        'response_format': {'type': 'json_object'},
        'messages': [
            {'content': """
            You are a helpful assistant
            
            Respond in JSON obeying the following JSON SCHEMA:
            {"additionalProperties": true, "properties": {"philosopher": {"title": "Philosopher", "type": "string"}, "meaning": {"title": "Meaning", "type": "string"}, "bullshit_level": {"default": 0.5, "title": "Bullshit Level", "type": "number"}}, "required": ["philosopher", "meaning"], "title": "MyResponseObject", "type": "object"}
            """, 'role': 'system'},
            {'content': """
            What is the meaning of life according to Douglas Adams?
            """, 'role': 'user'}
        ],
        'model': 'gpt-3.5-turbo'
    }
    completion = OpenAI().chat.completions.create(**args)
    content = json.loads(completion.choices[0].message.content)
    assert content['philosopher'] == "Douglas Adams"
