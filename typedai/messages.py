from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam


def System(content: str, name: str = None) -> ChatCompletionSystemMessageParam:
    rtn = dict(content=content, role="system")
    if name:
        rtn["name"] = name
    return rtn


def User(content: str, name: str = None) -> ChatCompletionUserMessageParam:
    rtn = dict(content=content, role="user")
    if name:
        rtn["name"] = name
    return rtn
