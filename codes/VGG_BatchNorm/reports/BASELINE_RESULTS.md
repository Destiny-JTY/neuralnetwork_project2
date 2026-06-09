# CIFAR-10 Baseline Results

## Model

`BaselineCNN` uses three convolution blocks:

```text
Input 3x32x32
Conv2d(3, 32, 3x3) -> ReLU -> MaxPool2d
Conv2d(32, 64, 3x3) -> ReLU -> MaxPool2d
Conv2d(64, 128, 3x3) -> ReLU -> MaxPool2d
Linear(2048, 256) -> ReLU -> Linear(256, 10)
```

Total trainable parameters: **620,362**.

## Training Setup

```text
Dataset: CIFAR-10 (50,000 train / 10,000 test)
Epochs: 10
Batch size: 128
Loss: CrossEntropyLoss
Optimizer: Adam
Learning rate: 0.001
Device used for this run: CPU
Random seed: 2020
```

Command:

```bash
python VGG_Loss_Landscape.py \
  --model baseline \
  --epochs 10 \
  --batch-size 128 \
  --optimizer adam \
  --lr 0.001 \
  --run-name baseline-adam-10ep
```

## Results

| Metric | Value |
| --- | ---: |
| Best epoch | 8 |
| Train loss at best epoch | 0.2977 |
| Train accuracy at best epoch | 89.55% |
| Test loss at best epoch | 0.7724 |
| Best test accuracy | 76.12% |
| Best test error | 23.88% |
| Average training time per epoch | 34.60 seconds |
| Average test time per epoch | 3.04 seconds |
| Total time for 10 epochs | 376.39 seconds |

Test accuracy stopped improving after epoch 8 while training accuracy continued
to increase. Test loss also rose in the final epochs, indicating overfitting.
This baseline therefore provides a useful reference for later regularization,
architecture, activation, and optimizer experiments.

Artifacts are stored in `reports/runs/baseline-adam-10ep/`.
