try:
    import tensorflow as tf

    if tf.version.VERSION < "2":
        raise ImportError(
            f"tfbm requires tensorflow >= 2 but found {tf.version.VERSION}"
        )
    del tf  # clean up namespace - just checking it's installed.
except ImportError as e:
    raise ImportError("tfbm requires tensorflow but no installation found.") from e
from .benchmarks import Benchmark, BenchmarkSpec, benchmark, run_benchmark

__all__ = [
    "Benchmark",
    "BenchmarkSpec",
    "benchmark",
    "run_benchmark",
]
