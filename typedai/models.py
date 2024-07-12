from __future__ import annotations

from typing import List, Any, Callable, Generic, TypeVar

from openai.types.chat import (ChatCompletion,
                               ChatCompletionMessage,
                               ChatCompletionMessageToolCall,
                               ChatCompletionToolMessageParam,
                               ChatCompletionChunk)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChoiceChunk, ChoiceDelta, ChoiceDeltaToolCall

T = TypeVar('T')


class TypedChatCompletion(ChatCompletion, Generic[T]):
    choices: List[TypedChoice[T]]


class TypedChoice(Choice, Generic[T]):
    message: TypedChatCompletionMessage[T]

    def tool_messages(self) -> List[ChatCompletionToolMessageParam]:
        return [tc.tool_message() for tc in self.message.tool_calls]


class TypedChatCompletionMessage(ChatCompletionMessage, Generic[T]):
    content: T
    tool_calls: List[TypedChatCompletionMessageToolCall]


class TypedChatCompletionMessageToolCall(ChatCompletionMessageToolCall):
    _fn: Callable

    def __call__(self):
        return self._fn()

    def tool_message(self, result: Any = None) -> ChatCompletionToolMessageParam:
        result = result or self()
        return ChatCompletionToolMessageParam(content=str(result), role="tool", tool_call_id=self.id)


class TypedChatCompletionChunk(ChatCompletionChunk):
    choices: List[TypedChoiceChunk]


class TypedChoiceChunk(ChoiceChunk):
    delta: TypedDeltaChoice

    def tool_messages(self) -> List[ChatCompletionToolMessageParam]:
        return [tc.tool_message() for tc in self.message.tool_calls]


class TypedDeltaChoice(ChoiceDelta):
    tool_calls: TypedChoiceDeltaToolCall


class TypedChoiceDeltaToolCall(ChoiceDeltaToolCall):
    _fn: Callable

    def __call__(self):
        return self._fn()

    def tool_message(self, result: Any = None) -> ChatCompletionToolMessageParam:
        result = result or self()
        return ChatCompletionToolMessageParam(content=str(result), role="tool", tool_call_id=self.id)
