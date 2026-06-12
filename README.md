# Neural Network Project 2

CIFAR-10 classification experiments with a configurable CNN baseline, VGG-A,
VGG-A with Batch Normalization, controlled ablations, optimizer comparisons,
and a BatchNorm loss-landscape study.

## Project Results

- Final baseline CNN: **1,422,218 parameters**
- Final checkpoint accuracy: **81.53%** in CPU re-evaluation
- VGG-A best test accuracy: **75.85%**
- VGG-A-BN best test accuracy: **81.69%**
- BatchNorm reduced the mean learning-rate loss-envelope width by **78.4%**

The final report is available in
[`PJ2_Final_Report.pdf`](PJ2_Final_Report.pdf).

## Repository Structure

```text
codes/VGG_BatchNorm/
  VGG_Loss_Landscape.py   # training entry point
  analyze_final_model.py  # checkpoint evaluation and visualizations
  compare_runs.py         # shared curves and loss-envelope plots
  aggregate_results.py    # aggregate run summaries
  models/
    baseline.py           # configurable baseline CNN
    vgg.py                # VGG-A and VGG-A-BN
  reports/
    runs/                 # experiment configurations and results
    comparisons/          # comparison figures
```

Large model checkpoints and the CIFAR-10 dataset are excluded from Git. The
final `best.pt` checkpoint is hosted on Hugging Face:

<https://huggingface.co/Destiny-JTY/nn_pj2>

## Setup

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Destiny-JTY/neuralnetwork_project2.git
cd neuralnetwork_project2/codes/VGG_BatchNorm

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a CUDA installation, install the PyTorch build appropriate for the local
CUDA version by following the official PyTorch instructions before installing
the remaining requirements.

## Download and Evaluate the Best Model

The analysis script reconstructs the model from the tracked `config.json`.
Download the checkpoint into the run directory expected by that configuration:

```bash
cd codes/VGG_BatchNorm

RUN_DIR="reports/runs/best-candidate-large-dropout-wd-ls-cosine-rerun"
mkdir -p "$RUN_DIR/models"

curl -L \
  "https://huggingface.co/Destiny-JTY/nn_pj2/resolve/main/best.pt?download=true" \
  -o "$RUN_DIR/models/best.pt"
```

Evaluate the checkpoint on the complete CIFAR-10 test set:

```bash
python analyze_final_model.py \
  --run-dir "$RUN_DIR" \
  --checkpoint "$RUN_DIR/models/best.pt" \
  --data-dir data \
  --device auto
```

On the first run, Torchvision downloads CIFAR-10 into `data/`. The command
automatically uses CUDA, Apple MPS, or CPU, in that order. A specific backend
can be selected with `--device cuda`, `--device mps`, or `--device cpu`.

The generated files are written to:

```text
reports/runs/best-candidate-large-dropout-wd-ls-cosine-rerun/analysis/
  analysis_summary.json
  confusion_matrix.csv
  confusion_matrix.png
  per_class_accuracy.csv
  correct_examples.png
  incorrect_examples.png
  first_layer_filters.png
  last_layer_filters.png
  feature_maps.png
```

The expected CPU re-evaluation result is approximately:

```text
test_accuracy=0.8153 test_error=0.1847
```

Small differences may occur across PyTorch versions and hardware backends.

### Download with the Hugging Face CLI

As an alternative to `curl`:

```bash
python -m pip install -U huggingface_hub

hf download Destiny-JTY/nn_pj2 best.pt \
  --local-dir "$RUN_DIR/models"
```

## Train the Final Baseline Configuration

```bash
cd codes/VGG_BatchNorm

python VGG_Loss_Landscape.py \
  --model baseline \
  --baseline-channels 64-128-256 \
  --fc-width 256 \
  --activation relu \
  --dropout 0.3 \
  --optimizer adam \
  --lr 0.001 \
  --weight-decay 0.0005 \
  --label-smoothing 0.1 \
  --scheduler cosine \
  --epochs 20 \
  --batch-size 128 \
  --seed 2020 \
  --device auto \
  --run-name best-candidate-large-dropout-wd-ls-cosine-rerun
```

Each run stores its configuration, metrics, plots, and checkpoints under
`reports/runs/<run-name>/`.

## Quick Smoke Test

Use a small subset to verify the installation:

```bash
python VGG_Loss_Landscape.py \
  --model baseline \
  --epochs 1 \
  --train-items 512 \
  --test-items 256 \
  --batch-size 128 \
  --device auto \
  --run-name smoke-baseline
```

## VGG-A and BatchNorm Comparison

```bash
python VGG_Loss_Landscape.py \
  --model vgg_a \
  --epochs 10 \
  --optimizer adam \
  --lr 0.001 \
  --batch-size 128 \
  --seed 2020 \
  --run-name vgg-a-adam-10ep

python VGG_Loss_Landscape.py \
  --model vgg_a_bn \
  --epochs 10 \
  --optimizer adam \
  --lr 0.001 \
  --batch-size 128 \
  --seed 2020 \
  --run-name vgg-a-bn-adam-10ep

python compare_runs.py curves \
  --runs vgg-a-adam-10ep vgg-a-bn-adam-10ep \
  --labels VGG-A VGG-A-BN \
  --output reports/comparisons/vgg_a_vs_bn_curves.png
```

Additional training, ablation, and loss-landscape commands are documented in
[`codes/VGG_BatchNorm/README.md`](codes/VGG_BatchNorm/README.md).

## Dataset

CIFAR-10 is downloaded automatically through Torchvision. The official dataset
page is <https://www.cs.toronto.edu/~kriz/cifar.html>.

## Notes

- Dataset files and model weights are intentionally ignored by Git.
- Do not use `--no-download` until CIFAR-10 has been downloaded successfully.
- The reported experiments use batch size 128 and random seed 2020.
- The final checkpoint does not contain BatchNorm layers; BatchNorm is studied
  separately in the VGG-A/VGG-A-BN experiments.
