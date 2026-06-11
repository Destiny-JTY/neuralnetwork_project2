"""Create shared comparison figures from saved training runs."""

import argparse
import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = SCRIPT_DIR / "reports" / "runs"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reports" / "comparisons"


def read_metrics(run_dir):
    path = run_dir / "logs" / "metrics.csv"
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def metric_values(rows, field):
    return [float(row[field]) for row in rows]


def run_label(run_dir, label=None):
    return label or run_dir.name


def save_curve_comparison(run_dirs, labels, output_path):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for index, run_dir in enumerate(run_dirs):
        rows = read_metrics(run_dir)
        label = run_label(run_dir, labels[index] if labels else None)
        epochs = metric_values(rows, "epoch")
        axes[0].plot(epochs, metric_values(rows, "train_loss"), linestyle="--", label=f"{label} train")
        axes[0].plot(epochs, metric_values(rows, "test_loss"), label=f"{label} test")
        axes[1].plot(epochs, metric_values(rows, "train_accuracy"), linestyle="--", label=f"{label} train")
        axes[1].plot(epochs, metric_values(rows, "test_accuracy"), label=f"{label} test")

    axes[0].set(title="Loss Comparison", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Accuracy Comparison", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def read_step_loss(run_dir):
    path = run_dir / "logs" / "step_loss.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing step_loss.txt: {path}")
    return np.loadtxt(path, dtype=float)


def compute_loss_envelope(loss_runs):
    if not loss_runs:
        raise ValueError("At least one loss run is required.")
    shortest_run = min(len(run) for run in loss_runs)
    if shortest_run == 0:
        raise ValueError("Loss runs must not be empty.")
    aligned = np.asarray([run[:shortest_run] for run in loss_runs], dtype=float)
    return aligned.min(axis=0), aligned.max(axis=0)


def parse_group(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("groups must use label=run1,run2 syntax")
    label, runs = value.split("=", 1)
    run_names = [item.strip() for item in runs.split(",") if item.strip()]
    if not label.strip() or not run_names:
        raise argparse.ArgumentTypeError("groups must include a non-empty label and runs")
    return label.strip(), run_names


def save_loss_landscape(groups, runs_dir, output_path, max_steps=None):
    figure, axis = plt.subplots(figsize=(10, 5))
    rows = []
    for label, run_names in groups:
        runs = [read_step_loss(resolve_run_dir(run_name, runs_dir)) for run_name in run_names]
        min_curve, max_curve = compute_loss_envelope(runs)
        length = min(len(min_curve), len(max_curve))
        if max_steps is not None:
            length = min(length, max_steps)
        steps = np.arange(length)
        min_values = min_curve[:length]
        max_values = max_curve[:length]
        axis.plot(steps, min_values, linewidth=1, label=f"{label} min")
        axis.plot(steps, max_values, linewidth=1, label=f"{label} max")
        axis.fill_between(steps, min_values, max_values, alpha=0.2)
        rows.append(
            {
                "label": label,
                "runs": ";".join(run_names),
                "aligned_steps": length,
                "mean_envelope_width": float(np.mean(max_values - min_values)),
                "max_envelope_width": float(np.max(max_values - min_values)),
            }
        )

    axis.set(title="Loss Landscape Envelope", xlabel="Training step", ylabel="Loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    write_landscape_summary(rows, output_path.with_suffix(".csv"))


def write_landscape_summary(rows, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "label",
                "runs",
                "aligned_steps",
                "mean_envelope_width",
                "max_envelope_width",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def resolve_run_dir(run_name, runs_dir):
    path = Path(run_name).expanduser()
    if path.exists():
        return path.resolve()
    return (runs_dir / run_name).resolve()


def parse_curves_args(subparsers):
    parser = subparsers.add_parser("curves", help="plot shared loss/accuracy curves")
    parser.add_argument("--runs", nargs="+", required=True, help="run names or run directories")
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "run_curves.png")
    return parser


def parse_landscape_args(subparsers):
    parser = subparsers.add_parser("landscape", help="plot loss-envelope landscape comparison")
    parser.add_argument(
        "--group",
        action="append",
        type=parse_group,
        required=True,
        help="label=run1,run2 syntax; pass once per envelope",
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "bn_loss_landscape.png")
    parser.add_argument("--max-steps", type=int, default=None)
    return parser


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    parse_curves_args(subparsers)
    parse_landscape_args(subparsers)
    return parser.parse_args()


def main():
    args = parse_args()
    runs_dir = args.runs_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if args.command == "curves":
        run_dirs = [resolve_run_dir(run_name, runs_dir) for run_name in args.runs]
        if args.labels is not None and len(args.labels) != len(run_dirs):
            raise ValueError("--labels must have the same length as --runs")
        save_curve_comparison(run_dirs, args.labels, output_path)
        print(f"Wrote curve comparison to {output_path}")
    elif args.command == "landscape":
        save_loss_landscape(args.group, runs_dir, output_path, args.max_steps)
        print(f"Wrote loss landscape to {output_path}")
        print(f"Wrote landscape summary to {output_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
