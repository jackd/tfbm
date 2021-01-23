# Tensorflow Benchmarks: [tfbm](https://github.com/jackd/tfbm)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Basic tensorflow utilities and CLI.

## Installation

```bash
pip install git+https://github.com/jackd/tfbm.git
```

## Example Usage

The following demonstrates basic usage.

```python
import tensorflow as tf

from tfbm import Benchmark, benchmark


class MyBenchmark(tfbm.Benchmark):
    # all method benchmarks duplicated for each spec here.
    BENCHMARK_SPEC = [
        benchmark(device='cpu', xla_jit=True),
        benchmark(device='gpu', xla_jit=True),
    ]

    # will be benchmarked 2 times, once for each BENCHMARK_SPEC entry.
    @benchmark(kwargs=dict(size=1024))
    def foo(self, size):
        x, y = get_args(size)
        return tf.square(x) + y

    # `bar` will be benchmarked 4 times
    @benchmark(xla_jit=True)
    @benchmark(xla_jit=False)
    def bar(self):
        ...

    @benchmark
    def baz(self):
        ...


if __name__ == "__main__":
    import tfbm.cli

    tfbm.cli.main()
```

The above generates the following benchmarks:

- `benchmark_foo_xla_cpu`
- `benchmark_foo_xla_gpu`
- `benchmark_bar_xla_cpu`
- `benchmark_bar_xla_gpu`
- `benchmark_bar_cpu`
- `benchmark_bar_gpu`
- `benchmark_baz_xla_cpu`
- `benchmark_baz_xla_gpu`

See [examples/benchmark_matmul.py](examples/benchmark_matmul.py) for a more complete example.

## Command Line Interface

`tfbm` exposes a command line interface that can be used to display results for any file(s) defining `tf.test.Benchmark`s. Note `tfbm.Benchmark` extends `tf.test.Benchmark`.

```bash
python -m tfbm examples/benchmark_matmul.py --group_by=device,spec --style=markdown
```

Results for device=gpu,spec=None
Uniform results:

| run_id |    cls | device | iters |
|--------|--------|--------|-------|
|    NOW | Matmul |    gpu |    10 |

Varied results:

|                        test | wall_time (us) | max_mem_GPU_0_bfc (Mb) | max_mem_gpu_host_bfc (b) | xla_jit |
|-----------------------------|----------------|------------------------|--------------------------|---------|
|      matmul_unstack_xla_gpu |        343.680 |                  8.000 |                   49.000 |    True |
| matmul_manual_transpose_gpu |       1345.634 |                   12.0 |                    8.000 |   False |
|        matmul_transpose_gpu |       1397.848 |                   12.0 |                    8.000 |   False |
|           matmul_einsum_gpu |       1467.109 |                   12.0 |                    8.000 |   False |
|          matmul_unstack_gpu |      51267.385 |               4104.000 |                     12.0 |   False |

-----------------------------------------------------------------------------------
Results for device=gpu,spec=XL
Uniform results:

| run_id |    cls | device | iters | spec |
|--------|--------|--------|-------|------|
|    NOW | Matmul |    gpu |    10 |   XL |

Varied results:

|                           test | wall_time (ms) | max_mem_GPU_0_bfc (Mb) | max_mem_gpu_host_bfc (b) | xla_jit |
|--------------------------------|----------------|------------------------|--------------------------|---------|
|      matmul_unstack_XL_xla_gpu |          3.013 |                128.000 |                   49.000 |    True |
|           matmul_einsum_XL_gpu |         64.821 |                  192.0 |                    8.000 |   False |
| matmul_manual_transpose_XL_gpu |         64.941 |                  192.0 |                    8.000 |   False |
|        matmul_transpose_XL_gpu |         65.027 |                  192.0 |                    8.000 |   False |
|          matmul_unstack_XL_gpu |       2790.027 |             262272.000 |                     12.0 |   False |

-----------------------------------------------------------------------------------
Results for device=cpu,spec=None
Uniform results:

| run_id |    cls | device | iters |
|--------|--------|--------|-------|
|    NOW | Matmul |    cpu |    10 |

Varied results:

|                        test | wall_time (ms) | max_mem_cpu (Mb) | max_mem_gpu_host_bfc (b) | xla_jit |
|-----------------------------|----------------|------------------|--------------------------|---------|
|      matmul_unstack_xla_cpu |          4.453 |            8.000 |                     49.0 |    True |
|        matmul_transpose_cpu |          9.065 |             12.0 |                      --- |   False |
| matmul_manual_transpose_cpu |          9.083 |             12.0 |                      --- |   False |
|           matmul_einsum_cpu |           9.34 |             12.0 |                      --- |   False |
|          matmul_unstack_cpu |         19.925 |         4104.043 |                      --- |   False |

## Pre-commit

This package uses [pre-commit](https://pre-commit.com/) to ensure commits meet minimum criteria. To Install, use

```bash
pip install pre-commit
pre-commit install
```

This will ensure git hooks are run before each commit. While it is not advised to do so, you can skip these hooks with

```bash
git commit --no-verify -m "commit message"
```
