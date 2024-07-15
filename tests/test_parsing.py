import copy

import pytest
from typedai.errors import ContentParsingError
from typedai.messages import System, User


@pytest.mark.vcr
@pytest.fixture
def completion(typed_ai):
    return copy.deepcopy(typed_ai.completions.create(
        messages=[System(content="You are a helpful assistant"), User(content="what is 2+2")], response_type=int
    ))  # returned content should be '{"response": 4}'


def test_can_parse_none_without_error(completion):
    completion.choices[0].message.content = None
    assert completion.parse_content() is None


def test_parsing_error(completion):
    completion.choices[0].message.content = "bad json"
    with pytest.raises(ContentParsingError) as e:
        completion.parse_content()
    assert e.value.content == "bad json"
    assert str(e.value.error) == "Expecting value: line 1 column 1 (char 0)"