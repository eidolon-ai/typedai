from openai import OpenAI, AsyncOpenAI

from .completions import TypedCompletions, AsyncTypedCompletions


class TypedAI:
    _client: OpenAI

    def __init__(self, *args, **kwargs):
        self._client = OpenAI(*args, **kwargs)

    @property
    def completions(self) -> TypedCompletions:
        return TypedCompletions(self._client.completions)


class AsyncTypedAI:
    _client: AsyncOpenAI

    def __init__(self, *args, **kwargs):
        self._client = AsyncOpenAI(*args, **kwargs)

    @property
    def completions(self) -> AsyncTypedCompletions:
        return AsyncTypedCompletions(self._client.completions)
