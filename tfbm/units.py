from typing import Mapping, OrderedDict, Sequence, Tuple, TypeVar

import numpy as np

from tfbm.names import BYTES_PREFIX

N = TypeVar("N", int, float)

BYTE_UNITS = (
    (1024 ** 4, "Tb"),
    (1024 ** 3, "Gb"),
    (1024 ** 2, "Mb"),
    (1024, "Kb"),
    (1, "b"),
)

TIME_UNITS = (
    (1e0, "s"),
    (1e-3, "ms"),
    (1e-6, "us"),
)


def is_time(header: str):
    return "wall_time" in header


def is_bytes(header: str):
    return BYTES_PREFIX in header


def add_units(header, units):
    return f"{header} ({units})"


def rescale_column(header, values):
    if is_time(header):
        divisor, units = biggest_time_units(np.nanmin(values))
        header = add_units(header, units)
        values = values / divisor
    elif is_bytes(header):
        divisor, units = biggest_byte_units(np.nanmin(values))
        header = add_units(header, units)
        values = values / divisor
    return header, values


def biggest_units(value, units_list: Sequence[Tuple[N, str]]) -> Tuple[N, str]:

    if np.isnan(value):
        raise ValueError("NaN values not supported")
    for units in units_list:
        if value > units[0]:
            return units
    return units_list[-1]


def biggest_byte_units(num_bytes: int) -> Tuple[int, str]:
    return biggest_units(num_bytes, BYTE_UNITS)


def biggest_time_units(seconds: float) -> Tuple[float, str]:
    return biggest_units(seconds, TIME_UNITS)


def rescale(cols: Mapping) -> OrderedDict:
    """Rescale all columns."""
    out = OrderedDict()
    for k, v in cols.items():
        k, v = rescale_column(k, v)
        out[k] = v
    return out
