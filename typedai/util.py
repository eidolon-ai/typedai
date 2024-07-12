import json
from typing import Callable, get_type_hints, Tuple, Dict, Iterable

from openai.types import FunctionDefinition as FunctionDefinitionModel
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChoiceChunk
from openai.types.shared_params import FunctionDefinition
from pydantic import create_model, TypeAdapter

from .models import TypedChoice, TypedChoiceChunk


def snake_to_capital_case(snake_str):
    components = snake_str.split('_')
    return ''.join(x.capitalize() for x in components)


def callable_params_as_base_model(func: Callable) -> TypeAdapter:
    # todo, add typedai type wrapper that will flatten a single type hint into its base json schema
    type_hints = get_type_hints(func)
    return create_model(snake_to_capital_case(func.__name__ + "Model"), **{param: (typ, ...) for param, typ in type_hints.items()})


def execute_tool_call(tool_call: ChatCompletionMessageToolCall, function: Callable, validator: TypeAdapter):
    obj = json.loads(tool_call.function.arguments)
    validator.validate_python(obj)
    return function(**obj)


def transform_tools(tools: Iterable[Callable]) -> Dict[str, Tuple[Callable, TypeAdapter, FunctionDefinition]]:
    tools = tools or []
    functions: Dict[str, Tuple[Callable, TypeAdapter, FunctionDefinition]] = dict()
    for fn in tools:
        if not callable(fn):
            raise ValueError(f"Expected a callable function, got {fn}")
        param_model = callable_params_as_base_model(fn)
        # noinspection PyArgumentList
        functions[fn.__name__] = fn, param_model, FunctionDefinitionModel(
            name=fn.__name__,
            description=fn.__doc__,
            parameters=param_model.json_schema(),
        ).model_dump()
    return functions


def type_choice(choice: Choice, functions: dict[str, Tuple[Callable, TypeAdapter, FunctionDefinition]]) -> TypedChoice:
    dumped = choice.model_dump()
    for tc in dumped["message"]['tool_calls']:
        fn, validator, _ = functions[tc["function"]["name"]]
        tc["_fn"] = lambda: execute_tool_call(tc, fn, validator)
    return TypedChoice(**dumped)


def type_choice_chunk(choice: ChoiceChunk, functions: dict[str, Tuple[Callable, TypeAdapter, FunctionDefinition]]) -> TypedChoiceChunk:
    dumped = choice.model_dump()
    for tc in dumped["delta"]['tool_calls']:
        fn, validator, _ = functions[tc["function"]["name"]]
        tc["_fn"] = lambda: execute_tool_call(tc, fn, validator)
    return TypedChoiceChunk(**dumped)
