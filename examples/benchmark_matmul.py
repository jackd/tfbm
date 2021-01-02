"""
Example usage for benchmarking various `A @ B.T` matrix multiplication implementations.

```bash
python -m tfbm benchmark_matmul.py --group_by=device,spec --style=markdown
```
"""

import tensorflow as tf

from tfbm import Benchmark, benchmark


def matmul_transpose(x, y):
    return tf.matmul(x, y, transpose_b=True)


def matmul_einsum(x, y):
    return tf.einsum("ij,kj->ik", x, y)


def matmul_unstack(x, y):
    return tf.add_n([tf.linalg.matvec(x, yi) for yi in tf.unstack(y, axis=0)])


def matmul_manual_transpose(x, y):
    return tf.matmul(x, tf.transpose(y, (1, 0)))


def get_args(i=1024, j=1024, k=1024):
    return tf.random.normal((i, j)), tf.random.normal((k, j))


class MatmulBenchmark(Benchmark):
    BENCHMARK_SPEC = [
        benchmark(device="cpu"),
        benchmark(device="gpu"),
        benchmark(name="XL", args=(4096,) * 3, device="gpu"),
    ]

    @benchmark
    def matmul_transpose(self, *args):
        return matmul_transpose(*get_args(*args))

    @benchmark
    def matmul_manual_transpose(self, *args):
        return matmul_manual_transpose(*get_args(*args))

    @benchmark
    def matmul_einsum(self, *args):
        return matmul_einsum(*get_args(*args))

    @benchmark(xla_jit=True)
    @benchmark(xla_jit=False)
    def matmul_unstack(self, *args):
        return matmul_unstack(*get_args(*args))


if __name__ == "__main__":
    import tfbm.cli

    tfbm.cli.main()
