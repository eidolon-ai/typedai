import copy

import pytest
from typedai.errors import ContentParsingError, ToolArgumentParsingError
from typedai.messages import System, User
from typedai.models import HANDLE_ANY_ERROR, HANDLE_PARSE_ERROR, ALWAYS_RAISE


@pytest.fixture
def base_completion(typed_ai):
    return copy.deepcopy(typed_ai.completions.create(
        messages=[System(content="You are a helpful assistant"), User(content="what is 2+2")], response_type=int
    ))  # returned content should be '{"response": 4}'


@pytest.mark.vcr
def test_can_parse_none_without_error(base_completion):
    base_completion.choices[0].message.content = None
    assert base_completion.parse_content() is None


@pytest.mark.vcr
def test_parsing_error(base_completion):
    base_completion.choices[0].message.content = "bad json"
    with pytest.raises(ContentParsingError) as e:
        base_completion.parse_content()
    assert e.value.content == "bad json"
    assert str(e.value.error) == "Expecting value: line 1 column 1 (char 0)"


def add(a: int, b: int) -> int:
    raise ValueError("Bad At Math")


@pytest.fixture
def tool_completion(typed_ai):
    completion = typed_ai.completions.create(model="gpt-3.5-turbo", messages=[
        System(
            content="You are a helpful assistant. You are very bad at math so you use tools to perform math for you."),
        User(content="What is 2 + 2?")
    ], fn_tools=add, )
    assert completion.choices[0].message.tool_calls
    return copy.deepcopy(completion)


@pytest.mark.vcr
def test_any_handles_execution_error(tool_completion):
    tool_completion.build_messages(tool_error_handling=HANDLE_ANY_ERROR)
    assert tool_completion.build_messages(tool_error_handling=HANDLE_ANY_ERROR)[-1][
               'content'] == "Error during tool execution\nValueError: Bad At Math"


@pytest.mark.vcr
def test_any_handlies_parsing_error(tool_completion):
    tool_completion.choices[0].message.tool_calls[0].function.arguments = "bad json"
    err_content = tool_completion.build_messages(tool_error_handling=HANDLE_ANY_ERROR)[-1]['content']
    assert "Error parsing arguments" in err_content
    assert "Invalid JSON: expected value at line 1 column 1" in err_content


@pytest.mark.vcr
def test_parsing_raises_execution_error(tool_completion):
    with pytest.raises(ValueError) as e:
        tool_completion.build_messages(tool_error_handling=HANDLE_PARSE_ERROR)
    assert str(e.value) == "Bad At Math"


@pytest.mark.vcr
def test_parsing_handles_parsing_error(tool_completion):
    tool_completion.choices[0].message.tool_calls[0].function.arguments = "bad json"
    err_content = tool_completion.build_messages(tool_error_handling=HANDLE_PARSE_ERROR)[-1]['content']
    assert "Error parsing arguments" in err_content
    assert "Invalid JSON: expected value at line 1 column 1" in err_content


@pytest.mark.vcr
def test_none_raises_execution_error(tool_completion):
    with pytest.raises(ValueError) as e:
        tool_completion.build_messages(tool_error_handling=ALWAYS_RAISE)
    assert str(e.value) == "Bad At Math"


@pytest.mark.vcr
def test_none_raises_parsing_error(tool_completion):
    tool_completion.choices[0].message.tool_calls[0].function.arguments = "bad json"
    with pytest.raises(ToolArgumentParsingError) as e:
        tool_completion.build_messages(tool_error_handling=ALWAYS_RAISE)
    assert "Error occurred while validating tool call arguments" in str(e.value)
