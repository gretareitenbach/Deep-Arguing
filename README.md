# 🧠 DeepArguing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/)

An implementation of **Deep Arguing**, leveraging modern PyTorch-based neural architectures to integrate with case-based reasoning (CBR) and abstract argumentation. 

---

## 📌 Table of Contents
- [Features](#-features)
- [Installation](#-installation)
- [Folder Structure](#-folder-structure)
- [Usage](#-usage)
- [Experiments & Results](#-experiments--results)
- [Citation & License](#-citation--license)

---

## ✨ Features
- **Argumentation-based Neural Pipelines:** End-to-end framework applying gradual semantics inside deep learning.
- **Configurable Hyperparameters:** Highly flexible YAML-based configuration for easy sweeping across datasets.
- **Built-in Support:** Plug-and-play models for standard benchmarks (MNIST, Adult, CIFAR-10, etc.).

---

## 🚀 Installation

To use DeepArguing, we recommend setting up a virtual environment (Python 3.12+). Then, install the project in editable mode to install all required dependencies (including `torch`, `networkx`, and `scikit-learn`).

```bash
# Clone the repository

# Install in editable mode
pip install -e .
```

---

## 📁 Folder Structure

A quick overview of the essential directories in this repository:

```text
Deep-Arguing/
├── data/               # Raw and processed datasets (MNIST, Adult, CIFAR-10, etc.)
├── experiments/        # Jupyter notebooks demonstrating model runs across benchmarks
├── src/
│   ├── deeparguing/    # Core library (gradual semantics, aacbr, criterion, base_scores)
│   └── scripts/        # Utility scripts (pre-training, tuning, generation)
├── tests/              # Pytest suite for automated testing and verification
├── tuning/             # YAML configurations for datasets, models, and hyperparameters
└── visualizer/         # Web-based visualization UI for computing graphs/arguments
```

---

## 💻 Usage

DeepArguing includes an intuitive CLI that accepts your pipeline's YAML configs to easily assemble, train, and test models. 

Here is an example command to run a full training and testing pipeline on the **MNIST** dataset:

```bash
python src/deeparguing/cli/run.py \
  --config "tuning/mnist/data_mnist.yaml" "tuning/mnist/hyperparameters_mnist.yaml" "tuning/mnist/model_mnist.yaml" \
  --seed "0" \
  --log "info" \
  --run_train -lv \
  --run_test
```

### CLI Arguments Overview:
- `--config`: Paths to data, hyperparameter, and model YAML configs.
- `--seed`: Ensures reproducibility across runs.
- `--log`: Sets logging verbosity (e.g., `info`, `debug`).
- `--run_train`: Executes the training loop.
- `-lv` / `--log_validation`: Enables validation logging during training.
- `--run_test`: Evaluates the model on the test dataset after training.

---

## 📊 Experiments & Results

Detailed experimental pipelines, benchmarks, and results can be found in the `experiments/` directory.

Each dataset (e.g., MNIST, Adult, Glioma, Covertype) has an associated Jupyter Notebook outlining setup, pipeline construction, and standard results across various baseline classifiers (DT, KNN, NN) vs. the Argumentative reasoning baseline. 

For reproducing results in the paper, please refer to the respective configuration files in the `tuning/` directory alongside the corresponding notebook in `experiments/`.

---

## 📜 License

Distributed under the **MIT License**. See `setup.py` for full details.

---
