import argparse
import ast
from typing import Any, Tuple

PLOT_NONE = 0
PLOT_BEFORE = 1
PLOT_AFTER = 2
PLOT_BOTH = 3


def parse_command_line():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        nargs="+",
        type=str,
        help="Paths to model config files (space seperated)",
    )

    parser.add_argument(
        "--seed",
        type=parse_seed,
        default=[0],
        help="Torch seed(s). Examples: single value: 42, list of seeds: [1,2,3], or range of seeds: (0,10)",
    )

    parser.add_argument(
        "--disable_tqdm", "-dt", action="store_true", help="Disable tqdm"
    )

    parser.add_argument(
        "--run_test", "-rt", action="store_true", help="Run on the test set"
    )

    parser.add_argument("--plot_loss", "-pl", action="store_true", help="Plot the loss")

    parser.add_argument(
        "--plot_gradients", "-pgr", action="store_true", help="Plot the gradients"
    )

    parser.add_argument(
        "--plot_matrix",
        "-pm",
        type=int,
        choices=[PLOT_NONE, PLOT_BEFORE, PLOT_AFTER, PLOT_BOTH],
        default=0,
        help=(
            "Plot the casebase as a matrix heatmap\n"
            "  0 = plot nothing\n"
            "  1 = plot before training\n"
            "  2 = plot after training\n"
            "  3 = plot both before and after training"
        ),
    )
    parser.add_argument(
        "--plot_graph",
        "-pg",
        type=int,
        choices=[PLOT_NONE, PLOT_BEFORE, PLOT_AFTER, PLOT_BOTH],
        default=0,
        help=(
            "Plot the casebase as a graph\n"
            "  0 = plot nothing\n"
            "  1 = plot before training\n"
            "  2 = plot after training\n"
            "  3 = plot both before and after training"
        ),
    )

    return parser.parse_args()


def parse_seed(value: Any) -> list[float]:
    try:
        # Try parsing as a literal (e.g. list or tuple)
        parsed = ast.literal_eval(value)
        if isinstance(parsed, int):
            return [parsed]
        elif isinstance(parsed, list):
            if all(isinstance(x, int) for x in parsed):
                return parsed
            else:
                raise ValueError("List must contain only integers.")
        elif isinstance(parsed, tuple) and len(parsed) == 2:
            start, end = parsed
            if isinstance(start, int) and isinstance(end, int):
                if start <= end:
                    return list(range(start, end))
                else:
                    raise ValueError("Range start must be <= end.")
            else:
                raise ValueError("Range values must be integers.")
        else:
            raise ValueError("Unsupported format for --seed.")
    except (SyntaxError, ValueError) as e:
        raise argparse.ArgumentTypeError(f"Invalid --seed value: {value}. Error: {e}")


def plots(args: Any) -> Tuple[bool, bool, bool, bool]:
    plot_matrix_before = args.plot_matrix in [PLOT_BEFORE, PLOT_BOTH]
    plot_matrix_after = args.plot_matrix in [PLOT_AFTER, PLOT_BOTH]
    plot_graph_before = args.plot_graph in [PLOT_BEFORE, PLOT_BOTH]
    plot_graph_after = args.plot_graph in [PLOT_AFTER, PLOT_BOTH]

    return plot_matrix_before, plot_matrix_after, plot_graph_before, plot_graph_after
