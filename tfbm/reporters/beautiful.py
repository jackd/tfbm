import shutil
from typing import Any, Optional, OrderedDict, Sequence, Tuple

import beautifultable as bt
import numpy as np
from termcolor import colored

from tfbm.reporters.registry import register


def column_to_string(
    column: np.ndarray,
    min_color: str = "green",
    max_color: str = "red",
    template="{:.3f}",
    nan_str="---",
) -> np.ndarray:
    """
    Convert a column to a string column.

    Only makes changes to np.float64 arrays.
    """
    if column.dtype != np.float64:
        return column
    min_value = np.nanmin(column)
    max_value = np.nanmax(column)
    if min_value == max_value:
        # don't color anything if all values are the same
        min_indices = max_indices = []
    else:
        (min_indices,) = np.where(column == min_value)
        (max_indices,) = np.where(column == max_value)
    # don't make it a numpy array - colors are weird
    column = [nan_str if np.isnan(v) else template.format(v) for v in column]
    for min_index in min_indices:
        column[min_index] = colored(column[min_index], min_color)
    for max_index in max_indices:
        column[max_index] = colored(column[max_index], max_color)
    return column


def get_style(style: str):
    orig_style = style
    style = style.lower()
    if not style.startswith("style_"):
        style = f"style_{style}"
    style = style.upper()
    for candidate in bt.enums.Style:
        if style == candidate.name:
            return candidate
    raise ValueError(f"Invalid style {orig_style}")


def get_table(
    results: OrderedDict[str, np.ndarray], style: Optional[str] = None, **kwargs
) -> Tuple[Sequence[Tuple[str, Any]], bt.BeautifulTable]:
    headers = []
    columns = []
    uniform_items = []
    for header, column in results.items():
        c0 = column[0]
        if np.all(column == c0):
            uniform_items.append((header, c0))
        else:
            headers.append(header)
            if column.dtype == np.float64:
                column = column_to_string(column, **kwargs)
            columns.append(column)

    terminal_shape = shutil.get_terminal_size()
    table = bt.BeautifulTable(
        maxwidth=terminal_shape[0], default_alignment=bt.ALIGN_RIGHT
    )
    if style is not None:
        table.set_style(get_style(style))
    table.columns.header = headers
    for row in zip(*columns):
        table.rows.append(row)

    uniform_header, uniform_row = zip(*uniform_items)
    uniform_table = bt.BeautifulTable(
        maxwidth=terminal_shape[0], default_alignment=bt.ALIGN_RIGHT
    )
    if style is not None:
        uniform_table.set_style(get_style(style))
    uniform_table.columns.header = uniform_header
    uniform_table.rows.append(uniform_row)
    return uniform_table, table


@register("beautiful")
def report(items, group_labels, style: Optional[str] = None):
    for group, results in items:
        uniform_table, table = get_table(results, style=style)
        print("-" * table.maxwidth)
        name = ",".join(
            (f"{label}={value}" for label, value in zip(group_labels, group))
        )
        print(f"Results for {name}")
        print("Uniform results:")
        print(uniform_table)
        print("Varied results:")
        print(table)
