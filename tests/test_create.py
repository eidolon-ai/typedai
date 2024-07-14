from openai import BaseModel
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


class MyResponseObject(BaseModel):
    philosopher: str
    meaning: str
    bullshit_level: float = .5


def test_typed_completions(typed_ai):
    completion = typed_ai.completions.create(
        model="gpt-3.5-turbo",
        messages=[System(content="You are a helpful assistant"),
                  User(content="What is the meaning of life according to Douglas Adams?")],
        response_type=MyResponseObject,
    )
    my_response_object = completion.choices[0].message.content
    assert my_response_object.philosopher == "Douglas Adams"
    assert "42" in my_response_object.meaning


def add(a: int, b: int) -> int:
    return a + b


def test_typed_tools(typed_ai):
    completion = typed_ai.completions.create(
        model="gpt-3.5-turbo",
        messages=[System(
            content="You are a helpful assistant. You are very bad at math so you use tools to perform math for you."
        ), User(
            content="What is 2 + 2?"
        )],
        fn_tools=add,
    )
    assert completion.choices[0].message.content is None
    message = completion.choices[0].message.tool_calls[0].build_completion_param()
    assert message['tool_call_id']
    message['tool_call_id'] = "stable_id"
    assert message == {'content': '4', 'role': 'tool', 'tool_call_id': "stable_id"}


def subtract(a: int, b: int) -> int:
    return a - b


def test_continuing_tool_call(typed_ai):
    messages = [System(
        content="You are a helpful assistant. You are very bad at math so you use tools to perform math for you."
    ), User(
        content="What is 2 + 2?"
    )]
    completion = typed_ai.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        fn_tools=[add, subtract],
    )
    messages.extend(completion.choices[0].messages())
    completion2 = typed_ai.completions.create(messages=messages, model="gpt-3.5-turbo")
    assert "4" in completion2.choices[0].message.content


def test_int_response(typed_ai):
    completion = typed_ai.completions.create(
        model="gpt-4-turbo",
        messages=[System(content="You are a helpful assistant."), User(content="What is 2 + 2?")],
        response_type=int,
    )
    assert completion.choices[0].message.content == 4


def test_int_response_can_be_continued(typed_ai):
    messages = [System(content="You are a helpful assistant."), User(content="What is 2 + 2?")]
    completion = typed_ai.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        response_type=int,
    )
    messages.extend(completion.choices[0].messages())
    messages.append(User(content="now multipy that by 3"))
    completion = typed_ai.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        response_type=int,
    )
    assert completion.choices[0].message.content == 12
