from functools import partial
from typing import Iterable, Union, Callable, TypeVar, Type, Optional

from typedai.config import Config

from openai import Stream
from openai.resources.chat import Completions
from openai.types import ChatModel
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam, ChatCompletionChunk
from typedai.errors import ContentParsingError, CycleLimitExceeded

from .models import TypedChatCompletion, TypedStream, HANDLE_ANY_ERROR
from .util import transform_tools, optional_parser, require_parser

T = TypeVar('T')


class TypedCompletions:
    _completions: Completions
    default_model: Optional[str]

    def __init__(self, completions: Completions, default_model: Optional[str] = None):
        self._completions = completions
        self.default_model = default_model

    def create(
            self,
            messages: Iterable[ChatCompletionMessageParam],
            model: Optional[Union[str, ChatModel]] = None,
            fn_tools: Union[Iterable[Callable], Callable] = None,
            response_type: Type[T] = str,
            **kwargs
    ) -> TypedChatCompletion[Optional[T]]:
        model = model or self.default_model
        fn_tools: Iterable[Callable] = _clean_maybe_iterable(fn_tools)
        functions = transform_tools(fn_tools)
        parser = require_parser if kwargs.pop("_require_parser", False) else optional_parser
        chat_args = dict(stream=False, model=model, **kwargs)
        messages, deserializer = Config.transform_messages_fn(messages, response_type)
        completion: ChatCompletion = self._create(chat_args, fn_tools, functions, messages, response_type)
        typed_chat_completion = TypedChatCompletion[Optional[response_type]].model_validate(completion.model_dump())
        typed_chat_completion._parser = partial(parser, parser=deserializer)
        typed_chat_completion._functions = functions
        return typed_chat_completion

    def create_to_completion(
            self,
            messages: Iterable[ChatCompletionMessageParam],
            model: Optional[Union[str, ChatModel]] = None,
            fn_tools: Union[Iterable[Callable], Callable] = None,
            response_type: Type[T] = str,
            max_cycles: int = 8,
            **kwargs,
    ) -> TypedChatCompletion[T]:
        mem = []
        additional_messages = messages
        count = 0
        while additional_messages and count <= max_cycles:
            mem.extend(additional_messages)
            completion = self.create(mem, model, fn_tools, response_type, _require_parser=True, **kwargs)
            count += 1
            if completion.has_tool_calls():
                additional_messages = completion.build_messages(tool_error_handling=HANDLE_ANY_ERROR)
            else:
                try:
                    completion.parse_content()
                    return completion
                except ContentParsingError as e:
                    additional_messages = [e.message()]
        raise CycleLimitExceeded(f"Cycle limit ({max_cycles}) exceeded")

    def stream(
            self,
            messages: Iterable[ChatCompletionMessageParam],
            model: Optional[Union[str, ChatModel]] = None,
            fn_tools: Iterable[Callable] | Callable = None,
            response_type: Type[T] = str,
            **kwargs
    ) -> TypedStream[Optional[T]]:
        model = model or self.default_model
        fn_tools: Iterable[Callable] = _clean_maybe_iterable(fn_tools)
        functions = transform_tools(fn_tools)
        chat_args = dict(stream=True, model=model, **kwargs)
        messages, deserializer = Config.transform_messages_fn(messages, response_type)
        stream: Stream[ChatCompletionChunk] = self._create(chat_args, fn_tools, functions, messages, response_type)
        return TypedStream(stream, partial(optional_parser, parser=deserializer), functions)

    def _create(self, chat_args, fn_tools, functions, messages, response_type):
        if response_type is not str:
            chat_args["response_format"] = {"type": "json_object"}
        chat_args['messages'] = messages
        if fn_tools:
            extra_tools = [dict(type="function", function=fd) for _, _, fd in functions.values()]
            chat_args.setdefault("tools", []).extend(extra_tools)
        return self._completions.create(**chat_args)


def _clean_maybe_iterable(value):
    if value is None:
        return []
    elif isinstance(value, Iterable) and not isinstance(value, str):
        return value
    else:
        return [value]
