import importlib
import os
import sys
from typing import List, Sequence

import tensorflow as tf
from absl import app

from tfbm import cli


def find_benchmark_files(paths: Sequence[str]) -> List[str]:
    """
    Find benchmark files from the given root search locations.

    For each path, if the path is a file we add it to the returned list if it contains
    "benchmark" and ends in `.py`. If it's a directory we search it.

    Args:
        paths: sequence of start locations.

    Returns:
        full_paths: sequence of user/var expanded paths.
    """
    ret_paths = []
    for path in paths:
        path = os.path.expanduser(os.path.expandvars(path))
        if tf.io.gfile.isdir(path):
            for (directory, _, filenames) in tf.io.gfile.walk(path):
                for fn in filenames:
                    if "benchmark" in fn and fn.endswith(".py"):
                        ret_paths.append(os.path.join(directory, fn))
        else:
            ret_paths.append(path)
    return ret_paths


def import_from_path(path):
    """Import the specified python file as a module from path."""
    assert path.endswith(".py")
    # path = os.path.abspath(path)
    name = path[:-3]
    if name.startswith("./"):
        name = name[2:]
    name = name.replace("/", ".").replace("-", "_")

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


def main(argv=None):
    if argv is None:
        argv = sys.argv
    paths = find_benchmark_files(argv[1:])
    for p in paths:
        import_from_path(p)
    cli.run()


if __name__ == "__main__":
    app.run(main)
