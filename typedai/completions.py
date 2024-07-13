import copy
import json
from typing import Iterable, Union, Callable, cast, TypeVar, Type, Tuple, Dict

from typedai.config import Config

try:
    import jinja2
except ImportError:
    jinja2 = None
from openai import Stream, BaseModel
from openai.resources.chat import Completions
from openai.types import ChatModel, FunctionDefinition
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import TypeAdapter

from .models import TypedChatCompletion, TypedChatCompletionChunk
from .util import transform_tools, type_choice, type_choice_chunk

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
            **kwargs
    ) -> TypedChatCompletion[T]:
        tools = tools or []
        if tools and not isinstance(tools, Iterable):
            tools = [tools]
        functions = transform_tools(tools)
        args, deserializer = _completion_args(False, model, messages, tools, response_format, kwargs, functions)
        completion = cast(ChatCompletion, self._completions.create(**args))
        dumped = completion.model_dump()
        dumped['choices'] = [type_choice(c, response_format, deserializer, functions) for c in completion.choices]
        return TypedChatCompletion[response_format].model_validate(dumped)

    def stream(
            self,
            messages: Iterable[ChatCompletionMessageParam],
            model: Union[str, ChatModel],
            tools: Iterable[Callable] | Callable = None,
            response_format: Type[T] = str,
            **kwargs
    ) -> Stream[TypedChatCompletionChunk[T]]:
        tools = tools or []
        if tools and not isinstance(tools, Iterable):
            tools = [tools]
        functions = transform_tools(tools)
        args = _completion_args(True, model, messages, tools, response_format, kwargs, functions)
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


def _completion_args(stream, model, messages, tools, response_format: type[T], kwargs, functions: Dict[str, Tuple[Callable, BaseModel, FunctionDefinition]]) -> Tuple[dict, Callable[[str], T]]:
    kwargs = copy.deepcopy(kwargs)
    kwargs['stream'] = stream
    messages, deserializer = Config.transform_messages_fn(messages, response_format)
    if response_format is not str:
        kwargs["response_format"] = {"type": "json_object"}
    if tools:
        kwargs['tools'] = [dict(type="function", function=fd) for _, _, fd in functions.values()]
    kwargs['messages'] = messages
    kwargs['model'] = model
    return kwargs, deserializer
