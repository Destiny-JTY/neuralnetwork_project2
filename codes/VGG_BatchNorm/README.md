# CIFAR-10 CNN, VGG-A, and BatchNorm experiments

The training entry point is `VGG_Loss_Landscape.py`. It automatically selects
CUDA, Apple MPS, or CPU and stores every experiment in a separate run directory.

## Environment

Install the required packages in your Python environment:

```bash
cd codes/VGG_BatchNorm
python -m pip install -r requirements.txt
```

The archive included in the starter files is incomplete. On the first run,
leave downloading enabled so that Torchvision can replace it with a verified
copy. Do not use `--no-download` until the dataset is complete.

## Quick smoke test

Run the lightweight baseline for one epoch on a small subset:

```bash
python VGG_Loss_Landscape.py \
  --model baseline \
  --epochs 1 \
  --train-items 512 \
  --test-items 256 \
  --run-name smoke-baseline
```

## Baseline experiment

`BaselineCNN` contains three `Conv2d + ReLU + MaxPool2d` blocks and a
two-layer fully connected classifier. A reasonable first complete run is:

```bash
python VGG_Loss_Landscape.py \
  --model baseline \
  --epochs 20 \
  --optimizer adam \
  --lr 0.001 \
  --run-name baseline-adam
```

## VGG experiments

```bash
python VGG_Loss_Landscape.py --model vgg_a --epochs 20 --lr 0.001
python VGG_Loss_Landscape.py --model vgg_a_bn --epochs 20 --lr 0.001
```

Use `python VGG_Loss_Landscape.py --help` to see optimizer, scheduler, device,
dataset subset, and output options.

Each run creates:

```text
reports/runs/<run-name>/
  config.json
  model.txt
  summary.json
  figures/training_curves.png
  logs/history.json
  logs/metrics.csv
  logs/step_loss.txt
  logs/step_gradient_norm.txt
  models/best.pt
  models/last.pt
```

`step_loss.txt` is the input for the later loss-landscape comparison across
different learning rates. Checkpoints contain model, optimizer, scheduler,
metrics, epoch, and experiment configuration.

`summary.json` records the parameter count, per-epoch timing summary, best
train/test metrics, best test error, and paths to the saved model weights.

The completed 10-epoch baseline result is summarized in
`reports/BASELINE_RESULTS.md`.
