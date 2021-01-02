import datetime
import os
import tempfile
from typing import OrderedDict

from absl import app, flags

from tfbm import reporters
from tfbm import results as results_lib
from tfbm import units, utils

NOW = "NOW"
flags.DEFINE_string("benchmarks", default=".*", help="")
flags.DEFINE_string("id", default=None, help="Name to save with results.")
flags.DEFINE_string("save", default=None, help="Directory to save results to.")
flags.DEFINE_string("reporter", default="beautiful", help="Reporter name.")
flags.DEFINE_multi_string(
    "compare",
    default=[],
    help="directories previous runs have been saved to compare to.",
)
flags.DEFINE_boolean("now", default=True, help="Run benchmarks in positional args.")

flags.DEFINE_list("order_by", default=["wall_time"], help="Key to sort results by")
flags.DEFINE_list("group_by", default=["cls"], help="Fields to group by")
flags.DEFINE_list("drop", default=[], help="columns to drop")
flags.DEFINE_list(
    "leading", default=["run_id", "cls", "test", "wall_time"], help="leading columns"
)
flags.DEFINE_list("trailing", default=[], help="trailing columns")
flags.DEFINE_string("style", default=None, help="Style string for reporting.")

FLAGS = flags.FLAGS


def create_default_id() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def run():
    results = []
    if FLAGS.now:
        run_id = FLAGS.id

        save_dir = FLAGS.save
        if save_dir is None:
            with tempfile.TemporaryDirectory() as tmp_dir:
                if run_id is None:
                    run_id = "NOW"
                run = results_lib.Run(tmp_dir, run_id, FLAGS.benchmarks)
                run.load_results()
        else:
            if run_id is None:
                run_id = create_default_id()
            run = results_lib.Run(
                os.path.join(save_dir, run_id), run_id, FLAGS.benchmarks
            )
        results.extend(run.results)

    for cmp in FLAGS.compare:
        if cmp != NOW:
            results.extend(results_lib.Run.from_directory(cmp).results)

    results = [res.to_dict() for res in results]
    groups = utils.group_by(results, key=utils.item_getter(*FLAGS.group_by))

    def to_columns(dicts):
        cols = results_lib.to_columns(
            sorted(dicts, key=utils.item_getter(*FLAGS.order_by))
        )
        ordered_cols = OrderedDict()
        for k in FLAGS.leading:
            ordered_cols[k] = cols.pop(k)
        final_cols = [cols.pop(k) for k in FLAGS.trailing]
        for k in sorted(cols):
            ordered_cols[k] = cols[k]
        for k, col in zip(FLAGS.trailing, final_cols):
            ordered_cols[k] = col
        return units.rescale(ordered_cols)

    groups = tuple((k, to_columns(groups[k])) for k in groups)

    reporter = reporters.get(FLAGS.reporter)
    reporter(groups, FLAGS.group_by, style=FLAGS.style)


def app_main(argv=None):
    if argv is not None and len(argv) > 1:
        raise ValueError(f"Unrecognized command line args {argv[1:]}")
    run()


def main(argv=None):
    return app.run(app_main, argv=argv)


if __name__ == "__main__":
    main()
