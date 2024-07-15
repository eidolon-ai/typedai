from typing import Optional

from openai import OpenAI

from .completions import TypedCompletions


class TypedAI:
    client: OpenAI
    default_model: Optional[str]

    def __init__(self, client: OpenAI = None, default_model: Optional[str] = None):
        if client is None:
            client = OpenAI()
        self.client = client
        self.default_model = default_model

    @property
    def completions(self, default_model: Optional[str] = None) -> TypedCompletions:
        return TypedCompletions(self.client.chat.completions, default_model or self.default_model)


# class AsyncTypedAI:
#     _client: AsyncOpenAI
#
#     def __init__(self, *args, **kwargs):
#         self._client = AsyncOpenAI(*args, **kwargs)
#
#     @property
#     def completions(self) -> AsyncTypedCompletions:
#         return AsyncTypedCompletions(self._client.completions)
