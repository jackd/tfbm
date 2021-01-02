import functools
import itertools
from typing import Callable, Iterable, Mapping, Optional, Union

import tensorflow as tf
from tensorflow.core.protobuf import config_pb2  # pylint: disable=no-name-in-module

_BENCHMARK_SPECS = "_benchmark_specS"


def spec_suffix(device: bool, xla_jit: bool, separate_compiled_gradients: bool):
    """
    Get a unique name suffix for the given spec args.

    Consistent with that used in `tensorflow.compiler.tests.xla_test.Benchmark` but
    starts with `sep-grads` if `separate_compiled_gradients` is True.
    """
    parts = []
    if separate_compiled_gradients:
        parts.append("sep-grads")
    if xla_jit:
        parts.append("xla")
    parts.append(device)
    return "_".join(parts)


def run_benchmark(
    tf_bench,
    builder_fn,
    device: str = "cpu",
    xla_jit: bool = False,
    separate_compiled_gradients: bool = False,
    name: Optional[str] = None,
    extras: Optional[Mapping[str, Union[str, float]]] = None,
):
    """Build a graph and run benchmarks against it, with or without XLA.

    Largely copied from `tensorflow.compiler.tests.xla_test.Benchmark`.
    Differences are:
        - if provided, the `name` parameter is used as is without xla or device
            suffixes.
        - `extras` recevied "xla_jit" (as string), "device" items

    Args:
        tf_bench: An instance of tf.test.Benchmark, used to run the benchmark.
        builder_fn: A function that builds a graph when invoked, and returns
                (name, fetches), where name is the name of the test, and fetches
                is a list of tensors to fetch as output.
        xla_jit: If true compile with the XLA JIT, otherwise use regular TF.
        device: The tensorflow device to run on, e.g. "cpu", "gpu".
        separate_compiled_gradients: If true put each gradient subgraph into a
            separate compilation scope. This gives fine-grained control over which
            portions of the graph will be compiled as a single unit. Compiling
            gradients separately may yield better performance for some graphs.
            The scope is named based on the scope of the forward computation as well
            as the name of the gradients. As a result, the gradients will be compiled
            in a scope that is separate from both the forward computation, and from
            other gradients.
        extras: passed to `tf_bench.run_op_benchmark`.
    """
    extras = extras or {}
    extras["xla_jit"] = str(xla_jit)
    extras["device"] = device
    with tf.Graph().as_default():
        targets = []
        with tf.device(device):
            fetches = []
            jit_scope = tf.xla.experimental.jit_scope
            with jit_scope(
                compile_ops=xla_jit,
                separate_compiled_gradients=separate_compiled_gradients,
            ):
                fetches = tf.nest.flatten(builder_fn())

            # We only want to benchmark the operations themselves, and not the data
            # transfer of the result(s).    Non-compiled identity ops ensure XLA
            # doesn't know we're dropping the results, otherwise it might compile
            # away the entire computation.
            for fetch in fetches:
                targets.append(tf.identity(fetch).op)

        config = config_pb2.ConfigProto(allow_soft_placement=True)
        with tf.compat.v1.Session(config=config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            if not name:
                suffix = spec_suffix(
                    device=device,
                    xla_jit=xla_jit,
                    separate_compiled_gradients=separate_compiled_gradients,
                )
                name = f"{builder_fn.__name__}_{suffix}"
            tf_bench.run_op_benchmark(
                sess, targets, name=name, extras=extras,
            )


def _updated_bool(base: Optional[bool], update: Optional[bool]):
    return base if update is None else update


def _updated_dict(base: Mapping, other: Mapping):
    out = dict(base)
    out.update(other)
    return out


def _if_not_none(x, default):
    return default if x is None else x


class BenchmarkSpec:
    """
    Specification for a benchmark.

    See `tfbm.Benchmark` for example usage. Note `tfbm.benchmark` is a factory function
    to create `BenchmarkSpec` with the benefit of being able to be used without as a
    decorator without calling.

    ```python
    class MyBenchmark(tfbm.Benchmark):
        @benchmark  # same as @benchmark(), @BenchmarkSpec()
        def foo(self):
            ...
    ```

    Args:
        device: device on which to run, e.g. 'cpu', 'gpu'. Defaults to 'cpu'.
        xla_jit: whether or not to perform XLA just-in-time  compilation. Defaults to
            False.
        separate_compiled_gradients: something magic. Defaults to False.
        extras: included in benchmark results.
        name: name without arg suffix.
        args: used in builder_fn.
        kwargs: used in builder_fn.
    """

    def __init__(
        self,
        device: Optional[str] = None,  # default "cpu"
        xla_jit: Optional[bool] = None,  # default False
        separate_compiled_gradients: Optional[bool] = None,  # default False
        extras: Optional[Mapping] = None,
        name: Optional[str] = None,
        args: Iterable = (),
        kwargs: Optional[Mapping] = None,
    ):
        self.device = device
        self.xla_jit = xla_jit
        self.separate_compiled_gradients = separate_compiled_gradients
        self.extras = extras or {}
        self.name = name
        self.args = tuple(args)
        self.kwargs = kwargs or {}

    def with_updates(self, other: "BenchmarkSpec") -> "BenchmarkSpec":
        """Get a new `BenchmarkSpec` with `self` overriden by `other`."""
        if self.name and other.name:
            name = "_".join((self.name, other.name))
        else:
            name = other.name or self.name
        return BenchmarkSpec(
            device=other.device or self.device,
            xla_jit=_updated_bool(self.xla_jit, other.xla_jit),
            separate_compiled_gradients=_updated_bool(
                self.separate_compiled_gradients, other.separate_compiled_gradients
            ),
            args=self.args + other.args,
            kwargs=_updated_dict(self.kwargs, other.kwargs),
            extras=_updated_dict(self.extras, other.extras),
            name=name,
        )

    def benchmark_method(
        self, builder_method: Callable
    ) -> Callable[[tf.test.Benchmark], None]:
        """Get a benchmark function that can be called with a `tf.test.Benchmark`."""
        device = "cpu" if self.device is None else self.device
        xla_jit = _if_not_none(self.xla_jit, False)
        separate_compiled_gradients = _if_not_none(
            self.separate_compiled_gradients, False
        )
        suffix = spec_suffix(
            device=device,
            xla_jit=xla_jit,
            separate_compiled_gradients=separate_compiled_gradients,
        )
        if self.name:
            suffix = "_".join((self.name, suffix))
        name = "_".join((builder_method.__name__, suffix))

        def ret_fn(tf_bench):
            builder_fn = functools.partial(
                builder_method, tf_bench, *self.args, **self.kwargs
            )
            extras = self.extras
            if self.name is not None:
                extras = dict(extras)
                assert "spec" not in extras
                extras["spec"] = self.name
            return run_benchmark(
                tf_bench,
                builder_fn,
                device=device,
                xla_jit=xla_jit,
                separate_compiled_gradients=separate_compiled_gradients,
                extras=extras,
                name=name,
            )

        ret_fn.__name__ = f"benchmark_{name}"
        return ret_fn

    def __call__(self, method: Callable) -> Callable:
        """
        Mark a method for benchmarking with this spec.

        Unspecified args in this instance's constructor may be overriden by the defining
        class's `BENCHMARK_SPEC`.
        """
        if not hasattr(method, _BENCHMARK_SPECS):
            setattr(method, _BENCHMARK_SPECS, [])
        getattr(method, _BENCHMARK_SPECS).append(self)
        return method


def benchmark(
    name: Optional[Union[Callable, str]] = None,
    device: Optional[str] = None,
    xla_jit: Optional[bool] = None,
    separate_compiled_gradients: Optional[bool] = None,
    extras: Optional[Mapping] = None,
    args: Iterable = (),
    kwargs: Optional[Mapping] = None,
):
    """
    Factory for creating a `BenchmarkSpec` that can also be used as a decorator.

    ```python
    class MyBenchmark(tfbm.Benchmark):
        @benchmark  # same as @benchmark(), @BenchmarkSpec()
        def foo(self):
            ...
    ```

    See `tfbm.BenchmarkSpec` for more examples and arg description.
    """
    # support use as @benchmark
    if callable(name):
        return benchmark()(name)

    return BenchmarkSpec(
        name=name,
        device=device,
        xla_jit=xla_jit,
        separate_compiled_gradients=separate_compiled_gradients,
        args=args,
        kwargs=kwargs,
        extras=extras,
    )


class BenchmarkMetaClass(type(tf.test.Benchmark)):
    """
    MetaClass which generates `benchmark_x` for each decorated method `x`.

    See `tfbm.Benchmark`.
    """

    def __new__(cls, class_name, bases, dct):
        cls_specs = dct.get("BENCHMARK_SPEC")
        if cls_specs is None:
            raise ValueError(
                "Classes with BencharkMetaClass must have 'BENCHMARK_SPEC' attr."
            )
        if isinstance(cls_specs, BenchmarkSpec):
            cls_specs = [cls_specs]

        for v in tuple(dct.values()):
            fn_specs = getattr(v, _BENCHMARK_SPECS, None)
            if fn_specs is not None:
                assert all(isinstance(s, BenchmarkSpec) for s in fn_specs)
                specs = [
                    cls_spec.with_updates(fn_spec)
                    for cls_spec, fn_spec in itertools.product(cls_specs, fn_specs)
                ]
                for spec in specs:
                    bm_method = spec.benchmark_method(v)
                    bm_name = bm_method.__name__
                    if bm_name in dct:
                        raise KeyError(f"Duplicate benchmarks named {bm_name}")
                    dct[bm_name] = bm_method

        return super().__new__(cls, class_name, bases, dct)


class Benchmark(tf.test.Benchmark, metaclass=BenchmarkMetaClass):
    """
    Class for dynamically generating benchmarks.

    ```python
    class MyBenchmark(tfbm.Benchmark):
        @benchmark  # same as @benchmark(), @BenchmarkSpec()
        def cube(self):
            return tf.random.uniform((1024,)) ** 3

        @benchmark(xla_jit=True)
        def foo(self):
            x = tf.random.uniform((1024,))
            return x*2 + x

        # multiple can be used to produce multiple benchmarks.
        # names used in `run_benchmark` must be unique. Some name mangling is performed
        # in `BenchmarkSpec` based on `xla_jit`, `device` and
        # `separate_compiled_gradients`, but multiple runs with custom args will result
        # in duplicate names (which will throw an error at class instantiation time)
        # unless custom names are provided.
        @benchmark(args=(1024,), name='small')
        @benchmark(args=(8192,), name='large')
        def reduce_sum(self, size):
            return tf.reduce_sum(tf.random.uniform((size,)))
    ```

    BENCHMARK_SPEC can be used to duplicate specs across all benchmarked functions.

    ```python
    class MyBenchmark(tfbm.Benchmark):
        BENCHMARK_SPEC = [
            benchmark(device='cpu', xla_jit=True),
            benchmark(device='gpu', xla_jit=True),
        ]

        # will be benchmarked 2 times, once for each of the above
        @benchmark(kwargs=dict(size=1024))
        def foo(self, size):
            ...

        # `bar` will be benchmarked 4 times
        @benchmark(xla_jit=True)
        @benchmark(xla_jit=False)
        def bar(self):
            ...
    ```
    """

    BENCHMARK_SPEC = benchmark()  # can also be a list / tuple.
