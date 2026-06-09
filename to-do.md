# PJ2 Remaining Work

Deadline: **June 14, 2026, 23:59**

## Completed

- [x] Repair and organize the starter project.
- [x] Implement `VGG_A_BatchNorm`.
- [x] Add unified device selection, training, evaluation, logging, checkpoints,
      and plotting.
- [x] Implement and train a simple CIFAR-10 baseline CNN.
- [x] Record parameter count, epoch time, train/test loss, train/test accuracy,
      best test error, and best checkpoint.
- [x] Establish the Adam baseline: **76.12% test accuracy / 23.88% test error**.

## 1. Model Ablation Experiments

Use the current Adam baseline as the control. Keep the seed, data, epochs,
batch size, and optimizer fixed unless that factor is being tested.

### Implementation

- [ ] Make the baseline architecture configurable:
  - [ ] convolution channel widths, such as `16-32-64`, `32-64-128`,
        and `64-128-256`;
  - [ ] fully connected hidden width, such as `128`, `256`, and `512`;
  - [ ] activation: ReLU, LeakyReLU, and GELU;
  - [ ] regularization: weight decay, label smoothing, and Dropout;
  - [ ] optional BatchNorm or another required improvement component.
- [ ] Add CLI arguments for these model and loss settings.
- [ ] Save every experiment configuration in `config.json`.
- [ ] Add a script that aggregates run summaries into one CSV/table.

### Minimum Controlled Experiments

- [ ] **Width:** baseline channels versus one smaller or larger configuration.
- [ ] **Loss/regularization:** plain cross-entropy versus:
  - [ ] weight decay, or
  - [ ] label smoothing.
- [ ] **Activation:** ReLU versus LeakyReLU or GELU.
- [ ] **Additional component:** baseline versus BatchNorm or Dropout.
- [ ] Run at least 4-6 meaningful configurations in total.
- [ ] For each run, record:
  - [ ] parameter count;
  - [ ] average epoch time;
  - [ ] best train/test loss and accuracy;
  - [ ] best test error;
  - [ ] best checkpoint path.
- [ ] Create one comparison table and one comparison plot.
- [ ] Select the strongest architecture/regularization configuration.

## 2. Optimizer Experiments

Use the selected model from the ablation stage and keep all non-optimizer
settings fixed.

- [ ] Train with SGD + momentum.
- [ ] Train with Adam.
- [ ] Train with AdamW.
- [ ] Choose sensible learning rates for each optimizer.
- [ ] Compare at least one scheduler where useful:
  - [ ] no scheduler;
  - [ ] StepLR or cosine annealing.
- [ ] Record convergence speed, best test accuracy/error, and training time.
- [ ] Plot optimizer train/test curves on shared figures.
- [ ] Create an optimizer comparison table.
- [ ] Select the optimizer and scheduler for the final model.

## 3. Final CIFAR-10 Model

- [ ] Combine the best architecture, activation, regularization, optimizer,
      learning rate, and scheduler.
- [ ] Train the final model for enough epochs with a fixed random seed.
- [ ] Save the best and last checkpoints.
- [ ] Record final parameter count and training speed.
- [ ] Report final test loss, accuracy, and error.
- [ ] Implement and save:
  - [ ] confusion matrix;
  - [ ] per-class accuracy;
  - [ ] correctly and incorrectly classified examples;
  - [ ] first-layer convolution filters;
  - [ ] at least one feature-map visualization.
- [ ] Write a short interpretation of the main failure classes and learned
      visual features.

## 4. VGG-A Versus VGG-A-BN

`VGG_A_BatchNorm` is implemented, but the formal comparison is still pending.

- [ ] Verify both models with a small smoke test.
- [ ] Train VGG-A and VGG-A-BN using:
  - [ ] the same seed;
  - [ ] the same CIFAR-10 split;
  - [ ] the same optimizer and learning rate;
  - [ ] the same batch size and number of epochs;
  - [ ] the same initialization principle.
- [ ] Record train/test loss and accuracy for every epoch.
- [ ] Compare parameter count and epoch time.
- [ ] Plot both models on shared loss and accuracy figures.
- [ ] Analyze:
  - [ ] convergence speed;
  - [ ] final performance;
  - [ ] oscillation or stability;
  - [ ] train-test gap.

## 5. BN Loss Landscape

The helper functions `compute_loss_envelope()` and `plot_loss_landscape()`
exist, but the required multi-run experiment has not been completed.

- [ ] Choose and report the learning rates, for example:
  - [ ] `1e-4`;
  - [ ] `5e-4`;
  - [ ] `1e-3`;
  - [ ] `2e-3`.
- [ ] Train VGG-A once for every selected learning rate.
- [ ] Train VGG-A-BN once for every selected learning rate.
- [ ] Keep seed, batch order, epochs, optimizer, and other settings fixed.
- [ ] Save per-training-step losses for every run.
- [ ] Load all loss files and compute per-step:
  - [ ] `min_curve`;
  - [ ] `max_curve`.
- [ ] Plot the VGG-A loss envelope with `fill_between()`.
- [ ] Plot the VGG-A-BN loss envelope with `fill_between()`.
- [ ] Put BN and non-BN envelopes on the same figure.
- [ ] Compare envelope width and loss stability.
- [ ] Explain whether the result supports the claim that BN smooths the
      optimization landscape.

### Optional BN Analysis

- [ ] Plot gradient-norm curves using the recorded step gradient norms.
- [ ] Measure gradient predictiveness.
- [ ] Measure maximum gradient difference over distance.

## 6. Report and Submission

### Report

- [ ] Add name and student ID.
- [ ] Write the report sections:
  - [ ] Introduction and task overview;
  - [ ] Dataset and preprocessing;
  - [ ] Baseline architecture and results;
  - [ ] Architecture/loss/activation ablations;
  - [ ] Optimizer experiments;
  - [ ] Final CIFAR-10 model and visualizations;
  - [ ] VGG-A versus VGG-A-BN;
  - [ ] BN loss landscape;
  - [ ] Conclusions and limitations.
- [ ] Explain results instead of only listing numbers.
- [ ] Include model structures, experiment settings, tables, and figures.
- [ ] Check that every figure has a caption and is referenced in the text.

### Links and Files

- [ ] Push the latest code to GitHub.
- [ ] Upload final model weights to Google Drive or another storage service.
- [ ] Provide a dataset link rather than uploading CIFAR-10 to GitHub.
- [ ] Add the GitHub, dataset, and model-weight links to the report.
- [ ] Verify that all links are publicly accessible or shared correctly.
- [ ] Export one final PDF.
- [ ] Check the PDF for missing figures, broken references, and unreadable text.
- [ ] Submit the PDF before **June 14, 2026, 23:59**.

## Recommended Execution Order

1. Implement configurable baseline variants and the result aggregator.
2. Run the 4-6 controlled ablation experiments.
3. Run SGD, Adam, and AdamW comparisons on the selected model.
4. Train and visualize the final CIFAR-10 model.
5. Run the controlled VGG-A versus VGG-A-BN comparison.
6. Run the eight loss-landscape experiments.
7. Finish the report, upload weights, verify links, and submit.
