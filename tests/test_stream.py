import pytest
from openai import BaseModel
from typedai.messages import System, User


@pytest.mark.vcr
def test_string_completions(typed_ai):
    completion = typed_ai.completions.stream(
        model="gpt-3.5-turbo",
        messages=[
            System(content="You are a helpful assistant"),
            User(content="What is the capital of France?"),
        ],
    ).completion()
    assert "paris" in completion.choices[0].message.content.lower()


class MyResponseObject(BaseModel):
    philosopher: str
    meaning: str
    bullshit_level: float = 0.5


@pytest.mark.vcr
def test_typed_completions(typed_ai):
    completion = typed_ai.completions.stream(
        model="gpt-3.5-turbo",
        messages=[
            System(content="You are a helpful assistant"),
            User(content="What is the meaning of life according to Douglas Adams?"),
        ],
        response_type=MyResponseObject,
    ).completion()
    my_response_object = completion.parse_content()
    assert my_response_object.philosopher == "Douglas Adams"
    assert "42" in my_response_object.meaning


def add(a: int, b: int) -> int:
    return a + b


@pytest.mark.vcr
def test_typed_tools(typed_ai):
    completion = typed_ai.completions.stream(
        model="gpt-3.5-turbo",
        messages=[
            System(
                content="You are a helpful assistant. You are very bad at math so you use tools to perform math for you."
            ),
            User(content="What is 2 + 2?"),
        ],
        fn_tools=add,
    ).completion()
    assert completion.parse_content() is None
    message = completion.build_messages()[-1]
    assert message["tool_call_id"]
    message["tool_call_id"] = "stable_id"
    assert message == {"content": "4", "role": "tool", "tool_call_id": "stable_id"}


def subtract(a: int, b: int) -> int:
    return a - b


@pytest.mark.vcr
def test_continuing_tool_call(typed_ai):
    messages = [
        System(
            content="You are a helpful assistant. You are very bad at math so you use tools to perform math for you."
        ),
        User(content="What is 2 + 2?"),
    ]
    completion = typed_ai.completions.stream(
        model="gpt-3.5-turbo",
        messages=messages,
        fn_tools=[add, subtract],
    ).completion()
    messages.extend(completion.build_messages())
    completion2 = typed_ai.completions.stream(
        messages=messages, model="gpt-3.5-turbo"
    ).completion()
    assert "4" in completion2.parse_content()


@pytest.mark.vcr
def test_int_response(typed_ai):
    completion = typed_ai.completions.stream(
        model="gpt-4-turbo",
        messages=[
            System(content="You are a helpful assistant."),
            User(content="What is 2 + 2?"),
        ],
        response_type=int,
    ).completion()
    assert completion.parse_content() == 4


@pytest.mark.vcr
def test_int_response_can_be_continued(typed_ai):
    messages = [
        System(content="You are a helpful assistant."),
        User(content="What is 2 + 2?"),
    ]
    completion = typed_ai.completions.stream(
        model="gpt-4-turbo",
        messages=messages,
        response_type=int,
    ).completion()
    messages.extend(completion.build_messages())
    messages.append(User(content="now multipy that by 3"))
    completion = typed_ai.completions.stream(
        model="gpt-4-turbo",
        messages=messages,
        response_type=int,
    ).completion()
    assert completion.parse_content() == 12
