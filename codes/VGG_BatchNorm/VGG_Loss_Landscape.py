"""Train baseline and VGG-A models on CIFAR-10 and record statistics.

Example:
    python VGG_Loss_Landscape.py --model baseline --epochs 20
    python VGG_Loss_Landscape.py --model vgg_a --epochs 20
    python VGG_Loss_Landscape.py --model vgg_a_bn --epochs 20
"""

import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from data.loaders import get_cifar_loader
from models.baseline import BaselineCNN
from models.vgg import VGG_A, VGG_A_BatchNorm, get_number_of_parameters


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reports" / "runs"


def select_device(requested="auto"):
    """Select CUDA, Apple MPS, or CPU without assuming a fixed GPU index."""
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but CUDA is not available.")
        if device.type == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS was requested, but MPS is not available.")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_random_seeds(seed_value=0, device=None):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    device = torch.device(device or "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_accuracy(model, data_loader, device):
    """Return classification accuracy while preserving model train/eval mode."""
    was_training = model.training
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for inputs, targets in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            predictions = model(inputs).argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    model.train(was_training)
    return correct / total if total else 0.0


def evaluate(model, data_loader, criterion, device):
    """Evaluate average loss and accuracy for one data loader."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for inputs, targets in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += batch_size

    model.train(was_training)
    if total == 0:
        return {"loss": 0.0, "accuracy": 0.0}
    return {"loss": total_loss / total, "accuracy": correct / total}


def _classifier_gradient_norm(model):
    """Record the final classifier weight gradient used by the assignment."""
    classifier = getattr(model, "classifier", None)
    if classifier is None:
        return float("nan")

    for layer in reversed(classifier):
        weight = getattr(layer, "weight", None)
        if weight is not None and weight.grad is not None:
            return weight.grad.detach().norm().item()
    return float("nan")


def _save_checkpoint(path, model, optimizer, scheduler, epoch, metrics, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def _write_epoch_csv(path, history):
    fields = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        "learning_rate",
        "train_seconds",
        "test_seconds",
        "epoch_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index, epoch in enumerate(history["epoch"]):
            writer.writerow({field: history[field][index] for field in fields})


def plot_training_history(history, save_path):
    epochs = history["epoch"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="train")
    axes[0].plot(epochs, history["test_loss"], marker="o", label="test")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, history["train_accuracy"], marker="o", label="train")
    axes[1].plot(epochs, history["test_accuracy"], marker="o", label="test")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def train(
    model,
    optimizer,
    criterion,
    train_loader,
    test_loader,
    device,
    output_dir,
    scheduler=None,
    epochs_n=100,
    best_model_path=None,
    config=None,
    record_gradients=True,
):
    """Train a model and persist metrics, checkpoints, and figures."""
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    models_dir = output_dir / "models"
    logs_dir = output_dir / "logs"
    for directory in (figures_dir, models_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    best_model_path = Path(best_model_path or models_dir / "best.pt")
    config = dict(config or {})
    model.to(device)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
        "learning_rate": [],
        "train_seconds": [],
        "test_seconds": [],
        "epoch_seconds": [],
        "step_loss": [],
        "step_gradient_norm": [],
        "best_epoch": 0,
        "best_test_accuracy": -1.0,
        "best_test_error": 1.0,
        "best_test_loss": None,
    }

    for epoch in tqdm(range(1, epochs_n + 1), unit="epoch"):
        started_at = time.perf_counter()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()

            if record_gradients:
                history["step_gradient_norm"].append(
                    _classifier_gradient_norm(model)
                )
            history["step_loss"].append(loss.item())

            optimizer.step()

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += batch_size

        train_metrics = {
            "loss": running_loss / total if total else 0.0,
            "accuracy": correct / total if total else 0.0,
        }
        train_seconds = time.perf_counter() - started_at
        test_started_at = time.perf_counter()
        test_metrics = evaluate(model, test_loader, criterion, device)
        test_seconds = time.perf_counter() - test_started_at
        learning_rate = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["test_loss"].append(test_metrics["loss"])
        history["test_accuracy"].append(test_metrics["accuracy"])
        history["learning_rate"].append(learning_rate)
        history["train_seconds"].append(train_seconds)
        history["test_seconds"].append(test_seconds)
        history["epoch_seconds"].append(train_seconds + test_seconds)

        checkpoint_metrics = {
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
        }
        if test_metrics["accuracy"] > history["best_test_accuracy"]:
            history["best_test_accuracy"] = test_metrics["accuracy"]
            history["best_test_error"] = 1.0 - test_metrics["accuracy"]
            history["best_test_loss"] = test_metrics["loss"]
            history["best_epoch"] = epoch
            _save_checkpoint(
                best_model_path,
                model,
                optimizer,
                scheduler,
                epoch,
                checkpoint_metrics,
                config,
            )

        if scheduler is not None:
            scheduler.step()

        tqdm.write(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"test_loss={test_metrics['loss']:.4f} "
            f"test_acc={test_metrics['accuracy']:.4f}"
        )

    final_metrics = {
        "train_loss": history["train_loss"][-1],
        "train_accuracy": history["train_accuracy"][-1],
        "test_loss": history["test_loss"][-1],
        "test_accuracy": history["test_accuracy"][-1],
    }
    _save_checkpoint(
        models_dir / "last.pt",
        model,
        optimizer,
        scheduler,
        epochs_n,
        final_metrics,
        config,
    )

    with (logs_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    _write_epoch_csv(logs_dir / "metrics.csv", history)
    np.savetxt(logs_dir / "step_loss.txt", history["step_loss"], fmt="%.8f")
    if record_gradients:
        np.savetxt(
            logs_dir / "step_gradient_norm.txt",
            history["step_gradient_norm"],
            fmt="%.8f",
        )
    plot_training_history(history, figures_dir / "training_curves.png")

    best_index = history["best_epoch"] - 1
    summary = {
        "model": config.get("model"),
        "model_parameters": config.get("model_parameters"),
        "device": config.get("device"),
        "epochs_completed": epochs_n,
        "total_training_seconds": sum(history["epoch_seconds"]),
        "average_epoch_seconds": float(np.mean(history["epoch_seconds"])),
        "average_train_seconds": float(np.mean(history["train_seconds"])),
        "average_test_seconds": float(np.mean(history["test_seconds"])),
        "best_epoch": history["best_epoch"],
        "best_train_loss": history["train_loss"][best_index],
        "best_train_accuracy": history["train_accuracy"][best_index],
        "best_test_loss": history["best_test_loss"],
        "best_test_accuracy": history["best_test_accuracy"],
        "best_test_error": history["best_test_error"],
        "best_test_error_percent": history["best_test_error"] * 100.0,
        "best_model_path": str(best_model_path.resolve()),
        "last_model_path": str((models_dir / "last.pt").resolve()),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return history


def compute_loss_envelope(loss_runs):
    """Compute per-step minimum and maximum over equally aligned loss runs."""
    if not loss_runs:
        raise ValueError("At least one loss run is required.")
    shortest_run = min(len(run) for run in loss_runs)
    if shortest_run == 0:
        raise ValueError("Loss runs must not be empty.")
    aligned = np.asarray([run[:shortest_run] for run in loss_runs], dtype=float)
    return aligned.min(axis=0), aligned.max(axis=0)


def plot_loss_landscape(envelopes, save_path, max_steps=None):
    """Plot one or more loss envelopes.

    Args:
        envelopes: Mapping from label to ``(min_curve, max_curve)``.
    """
    figure, axis = plt.subplots(figsize=(10, 5))
    for label, (min_curve, max_curve) in envelopes.items():
        length = min(len(min_curve), len(max_curve))
        if max_steps is not None:
            length = min(length, max_steps)
        steps = np.arange(length)
        min_values = np.asarray(min_curve[:length])
        max_values = np.asarray(max_curve[:length])
        axis.plot(steps, min_values, linewidth=1, label=f"{label} min")
        axis.plot(steps, max_values, linewidth=1, label=f"{label} max")
        axis.fill_between(steps, min_values, max_values, alpha=0.2)

    axis.set(title="Loss Landscape", xlabel="Training step", ylabel="Loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_channels(value):
    try:
        channels = tuple(int(part) for part in value.replace(",", "-").split("-"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "channels must be integers separated by '-' or ','"
        ) from exc
    if len(channels) != 3 or any(channel < 1 for channel in channels):
        raise argparse.ArgumentTypeError(
            "--baseline-channels expects three positive widths, e.g. 32-64-128"
        )
    return channels


def build_model(model_name, args=None):
    if model_name == "baseline":
        if args is None:
            return BaselineCNN()
        return BaselineCNN(
            channels=args.baseline_channels,
            hidden_width=args.fc_width,
            activation=args.activation,
            use_batch_norm=args.baseline_batch_norm,
            dropout=args.dropout,
        )

    models = {
        "vgg_a": VGG_A,
        "vgg_a_bn": VGG_A_BatchNorm,
    }
    return models[model_name]()


def build_optimizer(name, parameters, learning_rate, weight_decay):
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    return torch.optim.Adam(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def build_scheduler(name, optimizer, epochs, step_size, gamma):
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
    return None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["baseline", "vgg_a", "vgg_a_bn"],
        default="baseline",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--optimizer", choices=["adam", "adamw", "sgd"], default="adam"
    )
    parser.add_argument(
        "--scheduler", choices=["none", "step", "cosine"], default="none"
    )
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--baseline-channels",
        type=parse_channels,
        default=parse_channels("32-64-128"),
        help="three baseline channel widths, such as 16-32-64",
    )
    parser.add_argument("--fc-width", type=int, default=256)
    parser.add_argument(
        "--activation", choices=["relu", "leaky_relu", "gelu"], default="relu"
    )
    parser.add_argument("--baseline-batch-norm", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-items", type=int, default=-1)
    parser.add_argument("--test-items", type=int, default=-1)
    parser.add_argument("--val-items", dest="test_items", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-record-gradients", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if args.fc_width < 1:
        parser.error("--fc-width must be positive")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in the range [0, 1)")
    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error("--label-smoothing must be in the range [0, 1)")
    return args


def main():
    args = parse_args()
    device = select_device(args.device)
    set_random_seeds(args.seed, device)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or (
        f"{args.model}_{args.optimizer}_lr{args.lr:g}_{timestamp}"
    )
    run_dir = args.output_dir.expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["baseline_channels"] = list(args.baseline_channels)
    config["data_dir"] = str(args.data_dir.expanduser().resolve())
    config["output_dir"] = str(args.output_dir.expanduser().resolve())
    config["run_name"] = run_name
    config["device"] = str(device)

    pin_memory = device.type == "cuda"
    train_loader = get_cifar_loader(
        root=args.data_dir,
        batch_size=args.batch_size,
        train=True,
        num_workers=args.num_workers,
        n_items=args.train_items,
        download=not args.no_download,
        pin_memory=pin_memory,
    )
    test_loader = get_cifar_loader(
        root=args.data_dir,
        batch_size=args.batch_size,
        train=False,
        shuffle=False,
        num_workers=args.num_workers,
        n_items=args.test_items,
        download=not args.no_download,
        pin_memory=pin_memory,
    )

    model = build_model(args.model, args)
    model_parameters = get_number_of_parameters(model)
    config["model_parameters"] = model_parameters
    config["training_samples"] = len(train_loader.dataset)
    config["test_samples"] = len(test_loader.dataset)
    with (run_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    (run_dir / "model.txt").write_text(f"{model}\n", encoding="utf-8")

    optimizer = build_optimizer(
        args.optimizer, model.parameters(), args.lr, args.weight_decay
    )
    scheduler = build_scheduler(
        args.scheduler, optimizer, args.epochs, args.step_size, args.gamma
    )

    print(f"Device: {device}")
    print(f"Model parameters: {model_parameters:,}")
    print(f"Training samples: {len(train_loader.dataset):,}")
    print(f"Test samples: {len(test_loader.dataset):,}")
    print(f"Outputs: {run_dir}")

    history = train(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(label_smoothing=args.label_smoothing),
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        output_dir=run_dir,
        scheduler=scheduler,
        epochs_n=args.epochs,
        config=config,
        record_gradients=not args.no_record_gradients,
    )
    print(
        f"Best test accuracy: {history['best_test_accuracy']:.4f}, "
        f"test error: {history['best_test_error']:.4f} "
        f"(epoch {history['best_epoch']})"
    )


if __name__ == "__main__":
    main()
