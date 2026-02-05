# SQL Neural Network

A neural network implemented **entirely in SQL** using dbt and Databricks. Train and run inference on MNIST digit classification without leaving your data warehouse.

## What's This?

This project proves that SQL can do more than just SELECT statements. It implements:

- **Forward pass** (matrix multiplication, ReLU, softmax)
- **Backpropagation** (gradient computation through all layers)
- **Mini-batch SGD** (stochastic gradient descent training)

All in pure SQL.

## Architecture

```
Input (784) → Linear+ReLU (128) → Linear+ReLU (64) → Linear (10) → Softmax
```

A simple 3-layer neural network for classifying 28×28 MNIST handwritten digits.

## Project Structure

| Directory | Description |
|-----------|-------------|
| `initial_nn/` | Reference PyTorch implementation for validation |
| `training/` | SQL-based neural network training with backprop |
| `inference/` | SQL-based forward pass for trained models |

## Quick Start

### Training

```bash
cd training
python scripts/init_weights.py
python scripts/prepare_training_data.py
python scripts/train.py --epochs 10 --batch-size 16
```

### Inference

```bash
cd inference
python scripts/export_weights.py
dbt seed
dbt run --select tag:databricks
```

## Requirements

- Python 3.8+
- dbt-databricks
- PyTorch (for weight initialization and verification)
- Access to a Databricks workspace

## Why?

Why not!

