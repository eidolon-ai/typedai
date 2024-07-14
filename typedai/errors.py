from typing import List, Union, Tuple

from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion import Choice, ChatCompletion
from typedai.messages import User
from typedai.models import TypedChoice


class ChoiceParsingError(Exception):
    choice: Choice
    error: Exception

    def __init__(self, choice: Choice, error: Exception):
        self.choice = choice
        self.error = error
        super().__init__(f"Error occurred while parsing choice: {error}")

    def __str__(self):
        return str(self.error)

    def __repr__(self):
        return f"ChoiceParsingError(choice={self.choice.__repr__()}, error={self.error.__repr__()})"

    def messages(self) -> List[ChatCompletionMessageParam]:
        return [self.choice.model_dump(), User("An error occurred while parsing your response: {self.error}")]


class CompletionParsingError(Exception):
    raw_completion: ChatCompletion
    choices: List[Union[ChoiceParsingError | TypedChoice]]
    errors: List[Tuple[int, ChoiceParsingError]]

    def __init__(self, completion: ChatCompletion, choices: List[ChoiceParsingError]):
        self.choices = choices
        self.errors = [(i, c) for i, c in enumerate(choices) if isinstance(c, ChoiceParsingError)]
        if len(self.errors) == 1:
            index, error = self.errors[0]
            super().__init__(f"Error occurred while parsing choice[{index}]: {self.errors[0]}")
        else:
            super().__init__(f"Multiple errors occurred while parsing choices {', '.join([str(i) for i, _ in self.errors])}")

    def __str__(self):
        return str(self.errors)

    def __repr__(self):
        return f"MultiChoiceParsingError(choices={self.errors.__repr__()})"

    def messages(self) -> List[ChatCompletionMessageParam]:
        return self.choices[0].messages()
