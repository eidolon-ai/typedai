from typing import Iterable, Union, Callable, cast, TypeVar, Type

from typedai.config import Config
from typedai.errors import ChoiceParsingError, CompletionParsingError

try:
    import jinja2
except ImportError:
    jinja2 = None
from openai import Stream
from openai.resources.chat import Completions
from openai.types import ChatModel
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam, ChatCompletionChunk

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
            fn_tools: Union[Iterable[Callable], Callable] = None,
            response_type: Type[T] = str,
            **kwargs
    ) -> TypedChatCompletion[T]:
        fn_tools: Iterable[Callable] = _clean_maybe_iterable(fn_tools)
        chat_args = dict(
            stream=False,
            model=model,
            **kwargs
        )
        if response_type is not str:
            chat_args["response_format"] = {"type": "json_object"}
        messages, deserializer = Config.transform_messages_fn(messages, response_type)
        chat_args['messages'] = messages
        functions = transform_tools(fn_tools)
        if fn_tools:
            chat_args.setdefault("tools", []).extend([dict(type="function", function=fd) for _, _, fd in functions.values()])
        completion = cast(ChatCompletion, self._completions.create(**chat_args))
        dumped = completion.model_dump(exclude={"choices"})
        dumped["choices"] = []
        error = None
        for choice in completion.choices:
            try:
                dumped["choices"].append(type_choice(choice, response_type, deserializer, functions))
            except ChoiceParsingError as e:
                error = e
                dumped["choices"].append(e)
        if error:
            raise CompletionParsingError(completion, dumped["choices"]) from error
        return TypedChatCompletion[response_type].model_validate(dumped)

    def stream(
            self,
            messages: Iterable[ChatCompletionMessageParam],
            model: Union[str, ChatModel],
            fn_tools: Iterable[Callable] | Callable = None,
            **kwargs
    ) -> Stream[TypedChatCompletionChunk]:
        fn_tools: Iterable[Callable] = _clean_maybe_iterable(fn_tools)
        functions = transform_tools(fn_tools)
        chat_args = dict(
            stream=True,
            model=model,
            messages=messages,
            **kwargs
        )
        if fn_tools:
            chat_args.setdefault("tools", []).extend([dict(type="function", function=fd) for _, _, fd in functions.values()])

        stream = cast(Stream[ChatCompletionChunk], self._completions.create(**chat_args))
        for chunk in stream:
            dumped = chunk.model_dump()
            dumped['choices'] = [type_choice_chunk(c, functions) for c in chunk.choices]
            yield TypedChatCompletionChunk.model_validate(dumped)


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


def _clean_maybe_iterable(value):
    if value is None:
        return []
    elif isinstance(value, Iterable) and not isinstance(value, str):
        return value
    else:
        return [value]
