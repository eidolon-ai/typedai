from __future__ import annotations

from collections import defaultdict
from typing import List, Callable, Generic, TypeVar, Literal, Iterable, Type, Tuple, Any, Optional

from openai import Stream, BaseModel
from openai.types import FunctionDefinition
from openai.types.chat import (ChatCompletion,
                               ChatCompletionMessage,
                               ChatCompletionMessageToolCall,
                               ChatCompletionToolMessageParam,
                               ChatCompletionAssistantMessageParam, ChatCompletionMessageParam, ChatCompletionChunk)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import Function
from typedai.errors import ToolArgumentParsingError, ContentParsingError

T = TypeVar('T')

ALWAYS_RAISE = "none"
HANDLE_PARSE_ERROR = "parsing"
HANDLE_ANY_ERROR = "any"
ToolErrorHandling = Literal[ALWAYS_RAISE, HANDLE_PARSE_ERROR, HANDLE_ANY_ERROR]


class TypedChatCompletion(ChatCompletion, Generic[T]):
    _parser: Callable[[Optional[str]], T]
    _functions: dict[str, Tuple[Callable, Type[BaseModel], Any]]

    def parse_content(self, choice: int = 0) -> T:
        choice_ = self.choices[choice]
        try:
            return self._parser(choice_.message.content)
        except Exception as e:
            raise ContentParsingError(choice_.message.content, e) from e

    def build_messages(self, choice: int = 0, tool_error_handling: ToolErrorHandling = HANDLE_PARSE_ERROR) -> List[ChatCompletionMessageParam]:
        return [self.choices[choice].message.model_dump(), *self.build_tool_completions(choice, tool_error_handling)]

    def has_tool_calls(self, choice: int = 0) -> bool:
        return bool(self.choices[choice].message.tool_calls)

    def build_tool_completions(self, choice: int = 0, tool_error_handling: ToolErrorHandling = HANDLE_PARSE_ERROR) -> List[ChatCompletionToolMessageParam]:
        acc = []
        for tc in self.choices[choice].message.tool_calls or []:
            try:
                result = self.execute_tool_call(tc)
                acc.append(ChatCompletionToolMessageParam(content=str(result), role="tool", tool_call_id=tc.id))
            except ToolArgumentParsingError as e:
                if tool_error_handling == HANDLE_ANY_ERROR:
                    acc.append(ChatCompletionToolMessageParam(content=str(e), role="tool", tool_call_id=tc.id))
                else:
                    raise e
            except Exception as e:
                if tool_error_handling == HANDLE_ANY_ERROR:
                    acc.append(ChatCompletionToolMessageParam(content=str(e), role="tool", tool_call_id=tc.id))
                else:
                    raise e
        return acc

    def execute_tool_call(self, tool_call: ChatCompletionMessageToolCall) -> Any:
        fn, parser, _ = self._functions[tool_call.function.name]
        try:
            parsed = parser.model_validate_json(tool_call.function.arguments)
        except Exception as e:
            raise ToolArgumentParsingError(e) from e
        return fn(**{k: v for k, v in parsed})


class TypedStream(Iterable[ChatCompletionChunk], Generic[T]):
    stream: Stream
    _seen: List[ChatCompletionChunk]
    _terminated: bool
    _parser: Callable[[Optional[str]], T]
    _functions: dict[str, Tuple[Callable, Type[BaseModel], FunctionDefinition], Any]

    def __init__(self, stream, parser: Callable[[Optional[str]], T], functions):
        self.stream = stream
        self._seen = []
        self._terminated = False
        self._parser = parser
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
        return self.completion(allow_partial_iteration).build_messages(tool_error_handling)

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

        completion = TypedChatCompletion(**dumped_chunk)
        completion._parser = self._parser
        completion._functions = self._functions
        return completion
