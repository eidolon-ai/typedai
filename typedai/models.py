from __future__ import annotations

from typing import List, Any, Callable, Generic, TypeVar, Optional

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

    def messages(self) -> List[ChatCompletionToolMessageParam]:
        return [self.message.model_dump(), *(tc.tool_message() for tc in self.message.tool_calls)]


class TypedChatCompletionMessage(ChatCompletionMessage, Generic[T]):
    content: Optional[T] = None
    tool_calls: List[TypedChatCompletionMessageToolCall]


class TypedChatCompletionMessageToolCall(ChatCompletionMessageToolCall):
    _fn: Callable

    def __call__(self):
        return self._fn()

    def tool_message(self, result: Any = None) -> ChatCompletionToolMessageParam:
        result = result or self()
        return ChatCompletionToolMessageParam(content=str(result), role="tool", tool_call_id=self.id)


class TypedChatCompletionChunk(ChatCompletionChunk, Generic[T]):
    choices: List[TypedChoiceChunk[T]]


class TypedChoiceChunk(ChoiceChunk, Generic[T]):
    delta: TypedDeltaChoice[T]

    def tool_messages(self) -> List[ChatCompletionToolMessageParam]:
        return [tc.tool_message() for tc in self.message.tool_calls]


class TypedDeltaChoice(ChoiceDelta, Generic[T]):
    content: Optional[T] = None
    tool_calls: TypedChoiceDeltaToolCall


class TypedChoiceDeltaToolCall(ChoiceDeltaToolCall):
    _fn: Callable

    def __call__(self):
        return self._fn()

    def tool_message(self, result: Any = None) -> ChatCompletionToolMessageParam:
        result = result or self()
        return ChatCompletionToolMessageParam(content=str(result), role="tool", tool_call_id=self.id)
