from __future__ import annotations

from collections import defaultdict
from functools import partial
from typing import List, Callable, Generic, TypeVar, Optional, Literal, Iterable, Type, Tuple, Any

from openai import Stream, BaseModel
from openai.types import FunctionDefinition
from openai.types.chat import (ChatCompletion,
                               ChatCompletionMessage,
                               ChatCompletionMessageToolCall,
                               ChatCompletionToolMessageParam,
                               ChatCompletionAssistantMessageParam, ChatCompletionMessageParam, ChatCompletionChunk)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import Function
from typedai.errors import ToolArgumentParsingError, ChoiceParsingError, CompletionParsingError
from typedai.util import execute_tool_call

T = TypeVar('T')

ALWAYS_RAISE = "none"
HANDLE_PARSE_ERROR = "parsing"
HANDLE_ANY_ERROR = "any"
ToolErrorHandling = Literal[ALWAYS_RAISE, HANDLE_PARSE_ERROR, HANDLE_ANY_ERROR]


class TypedChatCompletion(ChatCompletion, Generic[T]):
    choices: List[TypedChoice[T]]

    def messages(self, tool_error_handling: ToolErrorHandling = HANDLE_PARSE_ERROR) -> List[ChatCompletionMessageParam]:
        return self.choices[0].messages(tool_error_handling)


class TypedChoice(Choice, Generic[T]):
    message: TypedChatCompletionMessage[T]

    def messages(self, tool_error_handling: ToolErrorHandling = HANDLE_PARSE_ERROR) -> List[ChatCompletionMessageParam]:
        return [self.message.as_message_param(), *(tc.build_completion_param(tool_error_handling) for tc in self.message.tool_calls)]


class TypedChatCompletionMessage(ChatCompletionMessage, Generic[T]):
    raw_content: Optional[str] = None
    content: Optional[T] = None
    tool_calls: List[TypedChatCompletionMessageToolCall]

    def as_message_param(self) -> ChatCompletionAssistantMessageParam:
        dumped = self.model_dump(exclude={"raw_content", "content"})
        dumped["content"] = str(self.raw_content)
        if not dumped.get("tool_calls"):
            del dumped["tool_calls"]
        return ChatCompletionAssistantMessageParam(**dumped)


class TypedChatCompletionMessageToolCall(ChatCompletionMessageToolCall):
    _fn: Callable

    def __call__(self):
        return self._fn()

    def build_completion_param(self, tool_error_handling: ToolErrorHandling = HANDLE_PARSE_ERROR) -> ChatCompletionToolMessageParam:
        try:
            return ChatCompletionToolMessageParam(content=str(self()), role="tool", tool_call_id=self.id)
        except ToolArgumentParsingError as e:
            if tool_error_handling in {HANDLE_PARSE_ERROR, HANDLE_ANY_ERROR}:
                return ChatCompletionToolMessageParam(content=str(e), role="tool", tool_call_id=self.id)
            else:
                raise e
        except Exception as e:
            if tool_error_handling == HANDLE_ANY_ERROR:
                return ChatCompletionToolMessageParam(content=str(e), role="tool", tool_call_id=self.id)
            else:
                raise e


class TypedStream(Iterable[ChatCompletionChunk], Generic[T]):
    stream: Stream
    _seen: List[ChatCompletionChunk]
    _terminated: bool
    _response_format: Type[T]
    _deserializer: Callable[[str], T]
    _functions: dict[str, Tuple[Callable, Type[BaseModel], FunctionDefinition]]

    def __init__(self, stream, response_format: Type[T], deserializer: Callable[[str], T], functions):
        self.stream = stream
        self._seen = []
        self._terminated = False
        self._response_format = response_format
        self._deserializer = deserializer
        self._functions = functions

    def __next__(self) -> ChatCompletionChunk:
        try:
            next__ = self.stream.__next__()
            if not isinstance(next__, ChatCompletionChunk):
                raise ValueError(f"Expected ChatCompletionChunk, got {type(next__)}")
            self._seen.append(next__)
            return next__
        except StopIteration:
            self._terminated = True
            raise

    def __iter__(self) -> TypedStream:
        return self

    def __enter__(self) -> TypedStream:
        return self

    def __exit__(self, *args, **kwargs):
        self.stream.__exit__(*args, **kwargs)

    def messages(self, tool_error_handling: ToolErrorHandling = HANDLE_PARSE_ERROR, allow_partial_iteration=False) -> List[ChatCompletionMessageParam]:
        return self.completion(allow_partial_iteration).messages(tool_error_handling)

    def completion(self, allow_partial_iteration=False) -> TypedChatCompletion[T]:
        if not allow_partial_iteration and not self._terminated:
            for _ in self:
                pass
        if not self._seen:
            raise ValueError("No completions have been seen")
        choice_acc = [dict(content_acc=[], tool_calls={}, finish_reasons="stop") for _ in self._seen[0].choices]
        for chunk in self._seen:
            for i in range(len(chunk.choices)):
                delta = chunk.choices[i].delta
                acc = choice_acc[i]
                if delta.content:
                    acc["content_acc"].append(delta.content)
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        tc_acc = acc['tool_calls'].setdefault(tc.index, dict(id=tc.id, function=defaultdict(list), type="function"))
                        dumped = tc.function.model_dump(exclude={"index"}, exclude_none=True)
                        for k, v in dumped.items():
                            if isinstance(v, str):
                                tc_acc['function'][k].append(v)
                            else:
                                raise ValueError(f"Unexpected type {type(v)} for {k} in {dumped}")
                if chunk.choices[i].finish_reason:
                    acc["finish_reasons"] = chunk.choices[i].finish_reason
        choices = []
        for i in range(len(choice_acc)):
            acc = choice_acc[i]
            content = "".join(acc["content_acc"]) or None
            tool_calls = []
            for tc in acc["tool_calls"].values():
                function = Function.model_validate({k: "".join(sub_acc) for k, sub_acc in tc['function'].items()})
                tool_calls.append(ChatCompletionMessageToolCall(id=tc['id'], type=tc['type'], function=function))
            message = ChatCompletionMessage.model_validate(
                dict(content=content, role="assistant", tool_calls=tool_calls or None)
            )
            choices.append(Choice.model_validate(
                dict(finish_reason=acc["finish_reasons"], index=i, message=message)
            ))

        dumped_chunk = self._seen[-1].model_dump(exclude={"choices"})
        dumped_chunk["choices"] = choices
        dumped_chunk["object"] = "chat.completion"

        completion = ChatCompletion(**dumped_chunk)
        dumped = completion.model_dump(exclude={"choices"})

        error = None
        dumped["choices"] = []
        for choice in choices:
            try:
                dumped["choices"].append(type_choice(choice, self._response_format, self._deserializer, self._functions))
            except ChoiceParsingError as e:
                error = e
                dumped["choices"].append(e)
        if error:
            raise CompletionParsingError(completion, dumped["choices"]) from error
        return TypedChatCompletion[self._response_format].model_validate(dumped)


def type_choice(choice: Choice, response_format: Type[T], deserializer: Callable[[str], T], functions: dict[str, Tuple[Callable, Type[BaseModel], Any]]) -> TypedChoice[T]:
    try:
        dumped = choice.model_dump()
        if choice.message.content is not None:
            dumped["message"]["raw_content"] = choice.message.content
            dumped["message"]["content"] = deserializer(choice.message.content)
        dumped["message"]['tool_calls'] = dumped["message"]['tool_calls'] or []
        typed_choice = TypedChoice[response_format].model_validate(dumped)
        for tc in typed_choice.message.tool_calls:
            fn, validator, _ = functions[tc.function.name]
            tc._fn = partial(execute_tool_call, tc.function.arguments, function=fn, validator=validator)
        return typed_choice
    except Exception as e:
        raise ChoiceParsingError(choice, e) from e
