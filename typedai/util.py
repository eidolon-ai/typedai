from typing import Callable, get_type_hints, Tuple, Dict, Iterable, Type, Optional, TypeVar

from openai.types import FunctionDefinition
from pydantic import create_model, BaseModel


def snake_to_capital_case(snake_str):
    components = snake_str.split('_')
    return ''.join(x.capitalize() for x in components)


def callable_params_as_base_model(func: Callable) -> Type[BaseModel]:
    # todo, add typedai type wrapper that will flatten a single type hint into its base json schema
    type_hints = get_type_hints(func)
    params = {param: (typ, ...) for param, typ in type_hints.items() if param != "return"}
    return create_model(snake_to_capital_case(func.__name__ + "Model"), **params)


def transform_tools(tools: Iterable[Callable]) -> Dict[str, Tuple[Callable, Type[BaseModel], dict]]:
    functions: Dict[str, Tuple[Callable, Type[BaseModel], dict]] = {}
    for fn in tools:
        if not callable(fn):
            raise ValueError(f"Expected a callable function, got {fn}")
        param_model = callable_params_as_base_model(fn)
        # noinspection PyArgumentList
        functions[fn.__name__] = fn, param_model, FunctionDefinition(
            name=fn.__name__,
            description=fn.__doc__,
            parameters=param_model.model_json_schema(),
        ).model_dump()
    return functions


T = TypeVar('T')


def optional_parser(v: Optional[str], parser: Callable[[str], T]) -> Optional[T]:
    return parser(v) if v is not None else None


def required_parser(v: Optional[str], parser: Callable[[str], T]) -> T:
    if v is None:
        raise ValueError("Expected a value, got None")
    return parser(v)
