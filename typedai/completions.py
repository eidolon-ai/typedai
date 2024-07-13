import copy
import json
from typing import Iterable, Union, Callable, cast, TypeVar, Type

try:
    import jinja2
except ImportError:
    jinja2 = None
from openai import Stream
from openai.resources.chat import Completions
from openai.types import ChatModel
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import TypeAdapter

from .models import TypedChatCompletion, TypedChatCompletionChunk
from .util import transform_tools, type_choice, type_choice_chunk

DEFAULT_SCHEMA_TEMPLATE = "{content}\n\nRespond in JSON obeying the following JSON SCHEMA: \n{schema}"
T = TypeVar('T')


class TypedCompletions:
    _completions: Completions

    def __init__(self, completions: Completions):
        self._completions = completions

    def create(
            self,
            messages: Iterable[ChatCompletionMessageParam],
            model: Union[str, ChatModel],
            tools: Iterable[Callable] | Callable = None,
            response_format: Type[T] = str,
            schema_dump_args: dict = None,
            schema_template: str = DEFAULT_SCHEMA_TEMPLATE,
            **kwargs
    ) -> TypedChatCompletion[T]:
        functions = transform_tools(tools)
        args = _completion_args(False, model, messages, tools, schema_dump_args, response_format, schema_template, kwargs, functions)
        completion = cast(ChatCompletion, self._completions.create(**args))
        dumped = completion.model_dump()
        dumped['choices'] = [type_choice(c, response_format, functions) for c in completion.choices]
        return TypedChatCompletion[response_format].model_validate(dumped)

    def stream(
            self,
            messages: Iterable[ChatCompletionMessageParam],
            model: Union[str, ChatModel],
            tools: Iterable[Callable] | Callable = None,
            response_format: Type[T] = str,
            schema_dump_args: dict = None,
            schema_template: str = DEFAULT_SCHEMA_TEMPLATE,
            **kwargs
    ) -> Stream[TypedChatCompletionChunk[T]]:
        functions = transform_tools(tools)
        args = _completion_args(True, model, messages, tools, schema_dump_args, response_format, schema_template, kwargs, functions)
        for chunk in self._completions.create(**args):
            dumped = chunk.model_dump()
            dumped['choices'] = [[type_choice_chunk(c, response_format, functions) for c in chunk.choices]]
            yield TypedChatCompletionChunk[response_format](**dumped)


# class AsyncTypedCompletions:
#     _completions: AsyncCompletions
#
#     def __init__(self, completions: AsyncCompletions):
#         self._completions = completions
#
#     async def create(
#             self,
#             messages: Iterable[ChatCompletionMessageParam],
#             model: Union[str, ChatModel],
#             **kwargs
#     ) -> ChatCompletion:
#         ...
#
#     async def stream(
#             self,
#             messages: Iterable[ChatCompletionMessageParam],
#             model: Union[str, ChatModel],
#             **kwargs
#     ) -> AsyncStream[ChatCompletionChunk]:
#         ...


def _completion_args(stream, model, messages, tools, schema_dump_args, response_format, message_template: str, kwargs, functions):
    kwargs = copy.deepcopy(kwargs)
    kwargs['stream'] = stream
    if response_format is not str:
        kwargs["response_format"] = { "type": "json_object" }
        messages = [copy.deepcopy(m) for m in messages]
        json_schema = TypeAdapter(response_format).json_schema()
        schema_str = json.dumps(json_schema, **(schema_dump_args or {}))
        if not messages:
            raise ValueError("Messages must not be empty")
        elif messages[-1].get("role") != "user" or not messages[-1].get("content"):
            raise ValueError("Last message must be a user message")
        else:
            new_content = message_template.format(content=messages[-1]["content"], schema=schema_str)
            messages[-1]["content"] = new_content
    if tools and not isinstance(tools, Iterable):
        tools = [tools]
    if tools is not None:
        kwargs['tools'] = [fd for _, _, fd in functions.values()]
    kwargs['messages'] = messages
    kwargs['model'] = model
    return kwargs
