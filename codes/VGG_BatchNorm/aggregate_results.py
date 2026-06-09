"""Aggregate CIFAR-10 experiment summaries into a comparison CSV."""

import argparse
import csv
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = SCRIPT_DIR / "reports" / "runs"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "reports" / "run_comparison.csv"


FIELDS = [
    "run_name",
    "model",
    "baseline_channels",
    "fc_width",
    "activation",
    "baseline_batch_norm",
    "dropout",
    "optimizer",
    "lr",
    "weight_decay",
    "label_smoothing",
    "scheduler",
    "epochs_completed",
    "model_parameters",
    "training_samples",
    "test_samples",
    "average_epoch_seconds",
    "best_epoch",
    "best_train_loss",
    "best_train_accuracy",
    "best_test_loss",
    "best_test_accuracy",
    "best_test_error",
    "best_test_error_percent",
    "best_model_path",
]


def load_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def format_value(value):
    if isinstance(value, (list, tuple)):
        return "-".join(str(item) for item in value)
    return value


def collect_runs(runs_dir):
    rows = []
    for summary_path in sorted(runs_dir.glob("*/summary.json")):
        run_dir = summary_path.parent
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        summary = load_json(summary_path)
        config = load_json(config_path)
        row = {field: "" for field in FIELDS}
        row.update({field: format_value(config.get(field, "")) for field in FIELDS})
        row.update({field: format_value(summary.get(field, row[field])) for field in FIELDS})
        row["run_name"] = config.get("run_name", run_dir.name)
        rows.append(row)
    return rows


def write_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = collect_runs(args.runs_dir.expanduser())
    write_csv(rows, args.output.expanduser())
    print(f"Wrote {len(rows)} runs to {args.output.expanduser()}")


if __name__ == "__main__":
    main()
