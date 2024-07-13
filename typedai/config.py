import copy
import json
from textwrap import dedent
from typing import List

from pydantic import TypeAdapter


def default_response_format_to_schema_fn(response_format) -> str:
    json_schema = TypeAdapter(response_format).json_schema()
    return json.dumps(json_schema, **Config.default_json_dump_args)


def default_message_transform_fn(messages: List[dict], schema_str: str) -> List[dict]:
    if not messages:
        raise ValueError("Messages must not be empty")
    messages = [m for m in messages]
    messages[0] = copy.deepcopy(messages[0])
    system_message = messages[0]
    if system_message.get("role") != "system" or not system_message.get("content"):
        raise ValueError("First message must be a system message")
    new_content = Config.default_template.format(content=system_message["content"], schema=schema_str)
    system_message["content"] = new_content
    return messages


class Config:
    default_template = dedent("""\
    {content}
    
    Respond in JSON obeying the following JSON SCHEMA:
    {schema}""")

    default_json_dump_args = {}

    response_format_to_schema_fn = default_response_format_to_schema_fn
    transform_messages_fn = default_message_transform_fn

