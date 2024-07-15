from typing import Optional

from openai.types.chat import ChatCompletionMessageParam
from typedai.messages import User


class ContentParsingError(Exception):
    content: Optional[str]
    error: Exception

    def __init__(self, content: Optional[str], error: Exception):
        self.content = content
        self.error = error
        super().__init__(f"{type(error).__name__}: {error}")

    def message(self) -> ChatCompletionMessageParam:
        return User(f"An error occurred while parsing your response: {str(self.error)}")


class ToolArgumentParsingError(Exception):
    error: Exception

    def __init__(self, error: Exception):
        self.error = error
        super().__init__(
            f"Error occurred while validating tool call arguments: {error}"
        )


class CycleLimitExceeded(Exception):
    pass
