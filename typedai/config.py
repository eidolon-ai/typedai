import copy
import json
from textwrap import dedent
from typing import List, TypeVar, Tuple, Callable, Type, get_origin

from openai import BaseModel
from pydantic import TypeAdapter

T = TypeVar("T")


def _load_resp(s):
    return json.loads(s)["response"]


def default_message_transform_fn(
    messages: List[dict], output_format: Type[T]
) -> Tuple[List[dict], Callable[[str], T]]:
    if output_format is str:
        return messages, lambda x: x
    elif (
        get_origin(output_format) is dict
        or output_format is dict
        or (isinstance(output_format, type) and issubclass(output_format, BaseModel))
    ):
        adapter = TypeAdapter(output_format)
        json_schema = adapter.json_schema()
        deserializer = adapter.validate_json
    else:
        json_schema = dict(
            type="object",
            properties=dict(resopose=TypeAdapter(output_format).json_schema()),
            required=["response"],
        )
        deserializer = _load_resp
    if not messages:
        raise ValueError("Messages must not be empty")
    messages = [m for m in messages]
    messages[0] = copy.deepcopy(messages[0])
    system_message = messages[0]
    if system_message.get("role") != "system" or not system_message.get("content"):
        raise ValueError("First message must be a system message")
    new_content = Config.default_template.format(
        content=system_message["content"],
        schema=json.dumps(json_schema, **Config.default_json_dump_args),
    )
    system_message["content"] = new_content
    return messages, deserializer


class Config:
    default_template = dedent(
        """\
    {content}
    
    Respond in JSON obeying the following JSON SCHEMA:
    {schema}"""
    )

    default_json_dump_args = {}
    transform_messages_fn = default_message_transform_fn
