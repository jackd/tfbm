import json
import os
from typing import Dict, List, NamedTuple, Optional, OrderedDict, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.core.util import test_log_pb2  # pylint: disable=no-name-in-module
from tensorflow.python.platform.benchmark import (  # pylint: disable=no-name-in-module
    TEST_REPORTER_TEST_ENV,
    _run_benchmarks,
)

from tfbm.names import BYTES_PREFIX

LONG_BYTES_PREFIX = "allocator_maximum_num_bytes_"


def _double_value(x) -> Optional[float]:
    if x is None:
        return None
    return x.double_value


def _double_or_string_value(x) -> Union[float, str]:
    xs = str(x)
    if xs.startswith("double_value"):
        return x.double_value
    if xs.startswith("string_value"):
        return x.string_value
    raise ValueError(f"Input {x} has neither double or string values.")


class Metric(NamedTuple):
    name: str
    value: float
    min_value: Optional[float]
    max_value: Optional[float]

    @classmethod
    def from_proto(cls, proto) -> "Metric":
        min_value = proto.min_value
        if min_value is not None:
            min_value = min_value.double_value
        return Metric(
            proto.name,
            proto.value.double_value,
            _double_value(proto.min_value),
            _double_value(proto.max_value),
        )


class BenchmarkResult(NamedTuple):
    """Result of a single benchmark."""

    run_id: str
    cls: str
    test: str
    iters: int
    wall_time: float  # median over all iterations
    extras: OrderedDict[str, Union[float, str]]
    metrics: Tuple[Metric, ...]

    def to_dict(self) -> Dict[str, Union[str, float]]:
        dct = dict(
            run_id=self.run_id,
            cls=self.cls,
            test=self.test,
            iters=self.iters,
            wall_time=self.wall_time,
        )

        def update(value, key, prefix):
            if key in dct:
                key = f"{prefix}.{key}"
            assert key not in dct
            dct[key] = value

        for k, v in self.extras.items():
            if k.startswith(LONG_BYTES_PREFIX):
                k = k.replace(LONG_BYTES_PREFIX, BYTES_PREFIX)
                if k == BYTES_PREFIX:
                    # not sure why these occasionally pop up - no point reporting though
                    assert v == 0
                    continue
            update(v, k, "e")
        for metric in self.metrics:
            k = metric.name
            if metric.min_value and metric.max_value is None:
                update(metric.value, k, "m")
            else:
                for attr in ("value", "min_value", "max_value"):
                    dct[f"{k}.{attr}"] = getattr(metric, attr)
        return dct

    @classmethod
    def from_proto(cls, entry, run_id: str = "") -> "BenchmarkResult":
        cls_str, test = entry.name.split(".")
        # common trimming
        if cls_str.endswith("Benchmark"):
            cls_str = cls_str[: -len("Benchmark")]
        if test.startswith("benchmark_"):
            test = test[len("benchmark_") :]
        extras = OrderedDict()
        for k in sorted(entry.extras):
            extras[k] = _double_or_string_value(entry.extras[k])
        metrics = []
        for metric in entry.metrics:
            metrics.append(Metric.from_proto(metric))
        metrics = tuple(metrics)

        return BenchmarkResult(
            run_id=run_id,
            cls=cls_str,
            test=test,
            iters=entry.iters,
            wall_time=entry.wall_time,
            extras=extras,
            metrics=metrics,
        )


def to_columns(dicts: Sequence[BenchmarkResult]) -> Dict[str, np.ndarray]:
    keys = set()
    for d in dicts:
        keys.update(d)
    num_results = len(dicts)
    cols = {}
    for k in keys:
        col = [None] * num_results
        for i, d in enumerate(dicts):
            col[i] = d.get(k)

        valid = [v for v in col if v is not None]
        if all(isinstance(v, int) for v in valid):
            default_value = -1
            dtype = np.int64
        elif all(isinstance(v, float) for v in valid):
            default_value = np.nan
            dtype = np.float64
        elif all(isinstance(v, str) for v in valid):
            default_value = ""
            dtype = str
        else:
            print(valid)
            raise ValueError("Inconsistent dtypes")
        cols[k] = np.array(
            [default_value if v is None else v for v in col], dtype=dtype
        )
    return cols


class Run:
    """
    Results from a run of benchmarks.

    Args:
        root_dir: root directory to save to.
        run_id: run identifier.
        benchmarks: string matcher to filter searches by.
    """

    def __init__(self, root_dir: str, run_id: str, benchmarks: str):
        self._root_dir = root_dir
        self._run_id = run_id
        self._benchmarks = benchmarks
        self._results = None

    @classmethod
    def _config_path(cls, root_dir: str):
        return os.path.join(root_dir, "config.json")

    def load_results(self):
        if self._results is None:
            if not tf.io.gfile.exists(self.benchmarks_dir):
                self.generate()
            self._results = tuple(load_benchmarks(self.benchmarks_dir, self.run_id))

    @property
    def results(self) -> Sequence[BenchmarkResult]:
        self.load_results()
        return self._results

    @property
    def root_dir(self) -> str:
        return self._root_dir

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def benchmarks(self) -> str:
        return self._benchmarks

    @property
    def benchmarks_dir(self) -> str:
        return os.path.join(self.root_dir, "benchmarks")

    @property
    def config_path(self) -> str:
        return self._config_path(self.root_dir)

    def get_config(self):
        return dict(
            root_dir=self._root_dir, run_id=self.run_id, benchmarks=self.benchmarks,
        )

    @classmethod
    def from_config(cls, config):
        if isinstance(config, str):
            # must be a path to a config
            if config.endswith(".json"):
                if not tf.io.gfile.exists(config):
                    raise IOError(f"No config file found at {config}")
                with tf.io.gfile.GFile(config, "r") as fp:
                    config = json.load(fp)
            else:
                raise ValueError(f"configs must saved in json format, got {config}")
        return cls(**config)

    @classmethod
    def from_directory(cls, root_dir: str):
        config_path = cls._config_path(root_dir)
        return cls.from_config(config_path)

    def generate(self, overwrite=False):
        """Dump config and benchmark results to disk."""
        root_dir = self.root_dir

        for i, _ in enumerate(tf.io.gfile.walk(root_dir)):
            if i == 0:
                continue
            if overwrite:
                tf.io.gfile.rmtree(overwrite)
                break

            raise IOError(
                f"Files already exists at {root_dir} - consider using `overwrite`."
            )

        tf.io.gfile.makedirs(root_dir)
        tf.io.gfile.makedirs(self.benchmarks_dir)

        config = self.get_config()
        with tf.io.gfile.GFile(self.config_path, "w") as fp:
            json.dump(config, fp, indent=4, sort_keys=True)

        generate_benchmarks(self.benchmarks_dir, benchmarks=self.benchmarks)


def load_benchmarks(benchmarks_dir: str, run_id: str = "") -> List[BenchmarkResult]:
    """Load benchmarks saved under `benchmarks_dir`."""
    filenames = tf.io.gfile.listdir(benchmarks_dir)
    entries = test_log_pb2.BenchmarkEntries()
    for filename in filenames:
        path = os.path.join(benchmarks_dir, filename)
        with tf.io.gfile.GFile(path, "rb") as fp:
            entries.MergeFromString(fp.read())
    return [
        BenchmarkResult.from_proto(e, run_id=run_id)
        for e in entries.entry  # pylint: disable=no-member
    ]


def generate_benchmarks(benchmarks_dir: str, benchmarks: str = ".*"):
    """Generate benchmark files under `benchmarks_dir`."""
    orig_env = os.environ.get(TEST_REPORTER_TEST_ENV)
    os.environ[TEST_REPORTER_TEST_ENV] = f"{benchmarks_dir}/"
    _run_benchmarks(benchmarks)
    # reset environ state
    if orig_env is None:
        del os.environ[TEST_REPORTER_TEST_ENV]
    else:
        os.environ[TEST_REPORTER_TEST_ENV] = orig_env
