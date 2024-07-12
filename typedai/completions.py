from typing import Iterable, Union, Callable, cast

from openai import AsyncStream, Stream
from openai.resources.chat import Completions, AsyncCompletions
from openai.types import ChatModel
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessageParam

from .models import TypedChatCompletion, TypedChatCompletionChunk
from .util import transform_tools, type_choice


class TypedCompletions:
    _completions: Completions

    def __init__(self, completions: Completions):
        self._completions = completions

    def create(
            self,
            messages: Iterable[ChatCompletionMessageParam],
            model: Union[str, ChatModel],
            tools: Iterable[Callable] | Callable = None,
            **kwargs
    ) -> TypedChatCompletion:
        kwargs['stream'] = False
        if tools and not isinstance(tools, Iterable):
            tools = [tools]
        functions = transform_tools(tools)
        if tools is not None:
            kwargs['tools'] = [fd for _, _, fd in functions.values()]
        completion = cast(ChatCompletion, self._completions.create(messages=messages, model=model, **kwargs))
        dumped = completion.model_dump()
        dumped['choices'] = [[type_choice(c, functions) for c in completion.choices]]
        return TypedChatCompletion(**dumped)

    def stream(
            self,
            messages: Iterable[ChatCompletionMessageParam],
            model: Union[str, ChatModel],
            tools: Iterable[Callable] | Callable = None,
            **kwargs
    ) -> Stream[TypedChatCompletionChunk]:
        kwargs['stream'] = True
        if tools and not isinstance(tools, Iterable):
            tools = [tools]
        functions = transform_tools(tools)
        if tools is not None:
            kwargs['tools'] = [fd for _, _, fd in functions.values()]
        for chunk in self._completions.create(messages=messages, model=model, **kwargs):
            dumped = chunk.model_dump()
            dumped['choices'] = [[type_choice(c, functions) for c in chunk.choices]]
            yield TypedChatCompletionChunk(**dumped)


class AsyncTypedCompletions:
    _completions: AsyncCompletions

    def __init__(self, completions: AsyncCompletions):
        self._completions = completions

    async def create(
            self,
            messages: Iterable[ChatCompletionMessageParam],
            model: Union[str, ChatModel],
            **kwargs
    ) -> ChatCompletion:
        ...

    async def stream(
            self,
            messages: Iterable[ChatCompletionMessageParam],
            model: Union[str, ChatModel],
            **kwargs
    ) -> AsyncStream[ChatCompletionChunk]:
        ...

