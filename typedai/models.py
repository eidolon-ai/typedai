from __future__ import annotations

from typing import List, Any, Callable, Generic, TypeVar, Optional, Literal

from openai.types.chat import (ChatCompletion,
                               ChatCompletionMessage,
                               ChatCompletionMessageToolCall,
                               ChatCompletionToolMessageParam,
                               ChatCompletionChunk, ChatCompletionAssistantMessageParam, ChatCompletionMessageParam)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChoiceChunk, ChoiceDelta, ChoiceDeltaToolCall
from typedai.errors import ToolArgumentParsingError

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


class TypedChatCompletionChunk(ChatCompletionChunk):
    choices: List[TypedChoiceChunk]


class TypedChoiceChunk(ChoiceChunk):
    delta: TypedDeltaChoice

    def tool_messages(self) -> List[ChatCompletionToolMessageParam]:
        return [tc.build_completion_param() for tc in self.message.tool_calls]


class TypedDeltaChoice(ChoiceDelta):
    tool_calls: TypedChoiceDeltaToolCall


class TypedChoiceDeltaToolCall(ChoiceDeltaToolCall):
    _fn: Callable

    def __call__(self):
        return self._fn()

    def build_completion_param(self, result: Any = None) -> ChatCompletionToolMessageParam:
        result = result or self()
        return ChatCompletionToolMessageParam(content=str(result), role="tool", tool_call_id=self.id)
