from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion import Choice
from typedai.messages import User


class ChoiceParsingError(Exception):
    choice: Choice
    error: Exception

    def __init__(self, choice: Choice, error: Exception):
        self.choice = choice
        self.error = error
        super().__init__(f"{type(error).__name__}: {error}")

    def message(self) -> ChatCompletionMessageParam:
        return User(f"An error occurred while parsing your response: {str(self.error)}")


class ToolArgumentParsingError(Exception):
    error: Exception

    def __init__(self, error: Exception):
        self.error = error
        super().__init__(f"Error occurred while validating tool call arguments: {error}")