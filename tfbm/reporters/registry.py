from typing import Callable, Iterable, Optional, OrderedDict, Tuple

import numpy as np

_REPORT_REGISTRY = {}


Reporter = Callable[
    [Iterable[Tuple[Iterable[str], OrderedDict[str, np.ndarray]]], Optional[str]], None
]


def register(name: Optional[str]):
    """
    Register a reporter.

    Reporter report on grouped items with signature `(items, style)`. Each item is
    `(key, data)`, where `key` is an iterable of strings (e.g. (cls, test)) and `data`
    is an `OrderedDict` of column data.

    All column data is a numpy array or either string or float64. float64 columns
    may have `np.nan` values - potentially entirely nan values.

    Appropriate unit conversion / header manipulation will have been performed such that
    float values will typically be bigger than 1 - but hopefully not too much bigger.
    """

    def wrapped_fn(fn: Reporter):
        if not callable(fn):
            raise ValueError("Registered functions must be callable.")
        registered_name = name or fn.__name__
        if registered_name == "<lambda>":
            raise ValueError("Cannot register lambda functions")
        if registered_name is None:
            raise ValueError("Cannot register unnamed functions")
        if registered_name in _REPORT_REGISTRY:
            raise ValueError(
                f"Cannot register reporter {registered_name}: "
                "already exists in registry."
            )
        _REPORT_REGISTRY[registered_name] = fn
        return fn

    return wrapped_fn


def get(name: str) -> Callable:
    return _REPORT_REGISTRY[name]
