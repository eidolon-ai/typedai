# TypedAI
[![PyPI - Version](https://img.shields.io/pypi/v/eidolon-typedai)](https://pypi.org/project/eidolon-typedai/)
[![Tests - Status](https://img.shields.io/github/actions/workflow/status/eidolon-ai/typedai/test_python.yml?logo=github&label=Test%20Python)](https://github.com/eidolon-ai/typedai/actions/workflows/test_python.yml)

TypedAi is a Python library that simplifies the OpenAI chat completions api using type hints.

Think Typer / FastApi for LLM interactions.

The library has no dependencies other than openai and pydantic.

```pip install eidolon-typedai```

## Easily Define Output Format
Defining json schema by hand error-prone. With TypedAI, you can define the output format with python types that are 
transformed to JsonSchema.

TypedAI then stores those types with the response to be used later parsing.

```python
from typedai import TypedAI
from pydantic import BaseModel

class Response(BaseModel):
    philosopher: str
    meaning_of_life: str

typed_completion = TypedAI().completions.create(  # returns TypedChatCompletion[Response]
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the meaning of life according to Douglas Adams?"},
    ],
    response_type=Response
)
print(typed_completion.parse_content().meaning_of_life)  # 42
```
## IDE Type Hinting
Sure you can code without them. You can live on rice and beans too, but you sure don't want to.

![Alt text](https://raw.githubusercontent.com/eidolon-ai/typedai/main/resources/type_hints.png)

## Define tool calls with function signatures 

```python
from typedai import TypedAI
from typing import Literal

def get_meaning_of_life(philosopher: Literal["Douglas Adams"]) -> str:
    """Call this tool to get the meaning of life according to a philosopher."""
    if philosopher == "Douglas Adams":
        return "42"

completion_with_tool_calls = TypedAI().completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the meaning of life according to Douglas Adams?"},
    ],
    fn_tools=[get_meaning_of_life],
)
```

## Easily Construct Response Messages
When a llm completion has tool calls, you normally need to parse the response and execute the function. 

Since you defined your tools with functions, TypedAI can execute them and build the response messages for you.

```python
from typedai import TypedAI
from above_example import get_meaning_of_life

messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the meaning of life according to Douglas Adams?"},
]
completion_with_tool_calls = TypedAI().completions.create(
    model="gpt-3.5-turbo", messages=messages, fn_tools=[get_meaning_of_life]
)
if completion_with_tool_calls.has_tool_calls():
    messages.extend(completion_with_tool_calls.build_messages())
    completion_with_tool_calls = TypedAI().completions.create(
        model="gpt-3.5-turbo", messages=messages, fn_tools=[get_meaning_of_life]
    )
```

## Full Streaming Support (even with typed responses)

And this all works great with streaming completions too!

```python
from typedai import TypedAI
from above_example import Response


typed_stream = TypedAI().completions.stream(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the meaning of life according to Douglas Adams?"},
    ],
    response_type=Response,
)

for chunk in typed_stream:
    print(chunk)  # iterates ChatCompletionChunk just like you would expect

typed_stream.completion()  # aggregated TypedCompletion[Response] (with type hints!)
```

## Purely Additive
TypedAI is purely additive. You can use it as much or as little as you want. It doesn't change the way you use OpenAI's 
API, it just makes it easier.

The objects used are either the raw openai objects or a minimally altered TypedAI child object with some additional 
functionality. Even dumping the models will give you unchanged results.

Similarly, no parsing or validation is done 
without explicit method calls, so the additional functionality will never slow you down.

## Contact Us
Have a question or running into a problem? Reach out to us on [discord](https://discord.com/invite/6kVQrHpeqG) and let us know how we can help.
