"""Analyze a trained CIFAR-10 checkpoint for the final report."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from data.loaders import get_cifar_loader
from models.baseline import BaselineCNN
from models.vgg import VGG_A, VGG_A_BatchNorm, get_number_of_parameters


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = SCRIPT_DIR / "reports" / "runs" / "best-candidate-large-dropout"
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def select_device(requested="auto"):
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but CUDA is not available.")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def build_model(config):
    model_name = config["model"]
    if model_name == "baseline":
        return BaselineCNN(
            channels=tuple(config.get("baseline_channels", (32, 64, 128))),
            hidden_width=config.get("fc_width", 256),
            activation=config.get("activation", "relu"),
            use_batch_norm=config.get("baseline_batch_norm", False),
            dropout=config.get("dropout", 0.0),
        )
    if model_name == "vgg_a":
        return VGG_A()
    if model_name == "vgg_a_bn":
        return VGG_A_BatchNorm()
    raise ValueError(f"Unsupported model: {model_name}")


def denormalize(images):
    return (images.detach().cpu() * 0.5 + 0.5).clamp(0.0, 1.0)


def save_confusion_matrix(confusion, output_path):
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(confusion, cmap="Blues")
    axis.set_xticks(np.arange(len(CLASS_NAMES)), labels=CLASS_NAMES, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(CLASS_NAMES)), labels=CLASS_NAMES)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Confusion Matrix")
    for row in range(confusion.shape[0]):
        for column in range(confusion.shape[1]):
            value = int(confusion[row, column])
            if value:
                axis.text(column, row, value, ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_examples(examples, output_path, title, max_items=16):
    if not examples:
        return
    examples = examples[:max_items]
    columns = 4
    rows = int(np.ceil(len(examples) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 2.5, rows * 2.6))
    axes = np.asarray(axes).reshape(-1)
    for axis in axes:
        axis.axis("off")
    for axis, item in zip(axes, examples):
        image = denormalize(item["image"].unsqueeze(0))[0].permute(1, 2, 0).numpy()
        axis.imshow(image)
        axis.set_title(
            f"T: {CLASS_NAMES[item['target']]}\nP: {CLASS_NAMES[item['prediction']]}",
            fontsize=8,
        )
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def first_conv_layer(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            return module
    return None


def save_first_layer_filters(model, output_path, max_filters=32):
    layer = first_conv_layer(model)
    if layer is None:
        return
    weights = layer.weight.detach().cpu()[:max_filters]
    count = weights.shape[0]
    columns = 8
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 1.4, rows * 1.4))
    axes = np.asarray(axes).reshape(-1)
    for axis in axes:
        axis.axis("off")
    for axis, weight in zip(axes, weights):
        image = weight
        image = image - image.min()
        image = image / image.max().clamp(min=1e-8)
        axis.imshow(image.permute(1, 2, 0).numpy())
    figure.suptitle("First-layer convolution filters")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def capture_first_conv_features(model, images):
    layer = first_conv_layer(model)
    if layer is None:
        return None
    captured = {}

    def hook(_module, _inputs, output):
        captured["features"] = output.detach().cpu()

    handle = layer.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(images)
    finally:
        handle.remove()
    return captured.get("features")


def save_feature_maps(model, image, device, output_path, max_maps=32):
    features = capture_first_conv_features(model, image.unsqueeze(0).to(device))
    if features is None:
        return
    maps = features[0, :max_maps]
    count = maps.shape[0]
    columns = 8
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 1.4, rows * 1.4))
    axes = np.asarray(axes).reshape(-1)
    for axis in axes:
        axis.axis("off")
    for axis, feature_map in zip(axes, maps):
        values = feature_map.numpy()
        axis.imshow(values, cmap="viridis")
    figure.suptitle("First-conv feature maps")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def evaluate_and_collect(model, loader, device, max_examples):
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    correct_examples = []
    incorrect_examples = []
    first_image = None
    total_loss = 0.0
    total = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()

    model.eval()
    with torch.inference_mode():
        for inputs, targets in loader:
            if first_image is None:
                first_image = inputs[0].detach().cpu()
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            predictions = logits.argmax(dim=1)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            correct += (predictions == targets).sum().item()
            total += batch_size

            for image, target, prediction in zip(inputs.cpu(), targets.cpu(), predictions.cpu()):
                target_value = int(target.item())
                prediction_value = int(prediction.item())
                confusion[target_value, prediction_value] += 1
                item = {
                    "image": image,
                    "target": target_value,
                    "prediction": prediction_value,
                }
                if target_value == prediction_value and len(correct_examples) < max_examples:
                    correct_examples.append(item)
                elif target_value != prediction_value and len(incorrect_examples) < max_examples:
                    incorrect_examples.append(item)
    return {
        "confusion": confusion,
        "correct_examples": correct_examples,
        "incorrect_examples": incorrect_examples,
        "first_image": first_image,
        "loss": total_loss / total if total else 0.0,
        "accuracy": correct / total if total else 0.0,
        "total": total,
    }


def write_confusion_csv(confusion, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true/pred"] + CLASS_NAMES)
        for class_name, row in zip(CLASS_NAMES, confusion):
            writer.writerow([class_name] + [int(value) for value in row])


def write_per_class_accuracy(confusion, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["class", "correct", "total", "accuracy"])
        writer.writeheader()
        for class_name, row in zip(CLASS_NAMES, confusion):
            total = int(row.sum())
            correct = int(row[CLASS_NAMES.index(class_name)])
            writer.writerow(
                {
                    "class": class_name,
                    "correct": correct,
                    "total": total,
                    "accuracy": correct / total if total else 0.0,
                }
            )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--test-items", type=int, default=-1)
    parser.add_argument("--max-examples", type=int, default=16)
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    config = load_json(run_dir / "config.json")
    checkpoint_path = args.checkpoint or run_dir / "models" / "best.pt"
    output_dir = args.output_dir or run_dir / "analysis"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    data_dir = Path(config.get("data_dir", SCRIPT_DIR / "data"))
    loader = get_cifar_loader(
        root=data_dir,
        batch_size=args.batch_size,
        train=False,
        shuffle=False,
        num_workers=args.num_workers,
        n_items=args.test_items,
        download=not args.no_download,
        pin_memory=device.type == "cuda",
    )

    results = evaluate_and_collect(model, loader, device, args.max_examples)
    confusion = results["confusion"]

    write_confusion_csv(confusion, output_dir / "confusion_matrix.csv")
    write_per_class_accuracy(confusion, output_dir / "per_class_accuracy.csv")
    save_confusion_matrix(confusion, output_dir / "confusion_matrix.png")
    save_examples(results["correct_examples"], output_dir / "correct_examples.png", "Correct examples")
    save_examples(results["incorrect_examples"], output_dir / "incorrect_examples.png", "Incorrect examples")
    save_first_layer_filters(model, output_dir / "first_layer_filters.png")
    if results["first_image"] is not None:
        save_feature_maps(model, results["first_image"], device, output_dir / "feature_maps.png")

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
        "model": config["model"],
        "model_parameters": get_number_of_parameters(model),
        "test_samples": results["total"],
        "test_loss": results["loss"],
        "test_accuracy": results["accuracy"],
        "test_error": 1.0 - results["accuracy"],
        "output_dir": str(output_dir),
    }
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(f"Wrote final-model analysis to {output_dir}")
    print(f"test_accuracy={results['accuracy']:.4f} test_error={1.0 - results['accuracy']:.4f}")


if __name__ == "__main__":
    main()
