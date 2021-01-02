BYTES_PREFIX = "max_mem_"


def is_time(header: str):
    return "wall_time" in header


def is_bytes(header: str):
    return BYTES_PREFIX in header
