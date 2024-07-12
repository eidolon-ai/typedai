from __future__ import annotations

from functools import cached_property
from typing import List, Any, Callable

from openai.types.chat import (ChatCompletion,
                               ChatCompletionMessage,
                               ChatCompletionMessageToolCall,
                               ChatCompletionToolMessageParam,
                               ChatCompletionChunk)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChoiceChunk, ChoiceDelta, ChoiceDeltaToolCall


class TypedChatCompletion(ChatCompletion):
    choices: List[TypedChoice]


class TypedChoice(Choice):
    message: TypedChatCompletionMessage

    def tool_messages(self) -> List[ChatCompletionToolMessageParam]:
        return [tc.tool_message() for tc in self.message.tool_calls]


class TypedChatCompletionMessage(ChatCompletionMessage):
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
