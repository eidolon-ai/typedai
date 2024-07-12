# TypedAI

TypedAi is a Python library for integrating that simplifies the OpenAI chat completions api by using method signature to 
describe tools and make assistant tool call responses callable.

The interface is identical to the completions api except that it splits the streaming and non-streaming calls for
more accurate typing.

```pip install typedai```

## Tool Construction
The tool name is the function name, description comes from the function docstring, and the json schema is generated from 
parameter type hints.

## Tool Call Execution

When executing a tool call, the llm provided parameters are validated and then the provided function 
is executed. 

### Example

```python
from typedai import TypedAI
from typing import Literal
from pydantic import BaseModel

# response format can be described via pydantic model
class MyResponseObject(BaseModel):
    philosopher: str
    meaning: str
    bullshit_level: float = .5

# tool(s) described entirely by function signature
def meaning_of_life(philosopher: Literal["Douglas Adams"]) -> str:
    """Return the meaning of life according to a philosopher."""
    return "42"


messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the purpose of life?"}
]
completion = TypedAI().completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    tools=meaning_of_life,
    response_format=MyResponseObject
)
choice = completion.choices[0]
tool_call = choice.message.tool_calls[0]

# llm response tool_calls are executable
assert tool_call() == 42

# tool call messages can be constructed from a response choice
updated_messages = [*messages, choice.message, *choice.tool_messages()]
```


