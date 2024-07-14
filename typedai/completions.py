from typing import Iterable, Union, Callable, cast, TypeVar, Type, Dict, Tuple

from pydantic import BaseModel
from typedai.config import Config
from typedai.errors import ChoiceParsingError, CompletionParsingError

try:
    import jinja2
except ImportError:
    jinja2 = None
from openai import Stream
from openai.resources.chat import Completions
from openai.types import ChatModel, FunctionDefinition
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam, ChatCompletionChunk

from .models import TypedChatCompletion, type_choice, TypedStream
from .util import transform_tools

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
        functions = transform_tools(fn_tools)
        chat_args = dict(stream=False, model=model, **kwargs)
        messages, deserializer = Config.transform_messages_fn(messages, response_type)
        completion: ChatCompletion = self._create(chat_args, fn_tools, functions, messages, response_type)
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
            response_type: Type[T] = str,
            **kwargs
    ) -> TypedStream[T]:
        fn_tools: Iterable[Callable] = _clean_maybe_iterable(fn_tools)
        functions = transform_tools(fn_tools)
        chat_args = dict(stream=True, model=model, **kwargs)
        messages, deserializer = Config.transform_messages_fn(messages, response_type)
        stream: Stream[ChatCompletionChunk] = self._create(chat_args, fn_tools, functions, messages, response_type)
        return TypedStream(stream, response_type, deserializer, functions)

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
