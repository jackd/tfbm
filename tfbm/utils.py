from typing import Callable, Iterable, List, OrderedDict, TypeVar

K = TypeVar("K")
V = TypeVar("V")


def group_by(iterable: Iterable[V], key: Callable[[V], K]) -> OrderedDict[K, List[V]]:
    """
    Group entries by the given key function.

    Args:
        iterable: input entries.
        key: function that maps entries to group keys.

    Returns:
        `OrderedDict` with keys from `key` and values associated with the entries from
            `iterable` that mapped to it.
    """
    out = OrderedDict()
    for value in iterable:
        k = key(value)
        out.setdefault(k, []).append(value)
    return out


def item_getter(*args, default=None) -> Callable:
    """
    Get a function that gets multiple items.

    Similar to `operator.itemgetter` but always returns a tuple, and supports defaults.
    """

    def f(x):
        return tuple(x.get(arg, default) for arg in args)

    return f
