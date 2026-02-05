# SQL Neural Network Training

Train a neural network **entirely in SQL** using mini-batch SGD, dbt, and Databricks!

## Overview

This project implements backpropagation and stochastic gradient descent in pure SQL. The training uses mini-batches for efficient learning with noisy but unbiased gradient estimates.

### Architecture

- **Input**: 784 pixels (28x28 MNIST images, flattened)
- **Layer 1**: Linear(784 → 128) + ReLU
- **Layer 2**: Linear(128 → 64) + ReLU
- **Layer 3**: Linear(64 → 10) logits
- **Output**: Softmax + Cross-Entropy Loss

### Training Method: Mini-batch SGD

| Aspect | Value |
|--------|-------|
| Gradient type | Stochastic (mini-batch) |
| Default batch size | 16 |
| Updates per epoch | N (samples / batch_size) |
| Optimization | Combined single-query + direct SQL updates |

## Quick Start

```bash
cd training

# 1. Initialize random weights
python scripts/init_weights.py

# 2. Prepare training data (100 train, 20 val)
python scripts/prepare_training_data.py --train-samples 100 --val-samples 20

# 3. Train!
python scripts/train.py --epochs 10 --batch-size 16 --learning-rate 0.01 --shuffle
```

## Training Script Options

```bash
python scripts/train.py [OPTIONS]

Options:
  --epochs N           Number of training epochs (default: 10)
  --batch-size N       Mini-batch size (default: 16)
  --learning-rate F    Learning rate (default: 0.01)
  --shuffle            Randomize batch order each epoch
  --validate-every N   Run validation every N epochs (default: 1, 0=never)
  --early-stopping N   Stop if no improvement for N epochs (0=disabled)
  --init               Initialize random weights before training
  --verbose            Show detailed dbt output
```

## Example Output

```
======================================================================
  ⚡ SQL Neural Network - Mini-batch SGD Training
======================================================================
  Epochs:         10
  Batch size:     16
  Learning rate:  0.01
  Shuffle:        True

  Training samples: 100
  Batches per epoch: 7
  Updates per epoch: 7

──────────────────────────────────────────────────────────────────────
  Epoch 1/10: ....... loss=2.3456 acc=12.0% val_loss=2.4012 val_acc=10.0% (8.2s)
  Epoch 2/10: ....... loss=1.8234 acc=35.0% val_loss=1.9123 val_acc=30.0% (7.9s) ✓
  Epoch 3/10: ....... loss=1.2345 acc=58.0% val_loss=1.3456 val_acc=55.0% (8.1s) ✓
  ...
──────────────────────────────────────────────────────────────────────

  Total time: 82.3s (8.2s/epoch)
  Best validation loss: 0.8234
  Final: train_loss=0.7123 train_acc=78.0%

  Saved: training_history.csv, training_curves.png
```

## Project Structure

```
training/
├── dbt_project.yml              # dbt config
├── profiles.yml                 # Databricks connection
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── training_history.csv         # Generated: epoch metrics
├── training_curves.png          # Generated: loss/accuracy plots
│
├── macros/
│   └── nn_backward.sql          # Backpropagation macros (reference)
│
├── models/training/
│   ├── training_step_sgd.sql       # ⚡ Main: combined fwd+bwd+update
│   ├── validation_forward_pass.sql # Validation forward pass
│   ├── validation_softmax.sql      # Validation softmax
│   ├── validation_loss.sql         # Validation loss
│   ├── validation_accuracy.sql     # Validation metrics
│   ├── loss.sql                    # Training loss helper
│   ├── accuracy.sql                # Training accuracy helper
│   ├── predictions.sql             # Predictions helper
│   ├── epoch_summary.sql           # Combined metrics
│   ├── training_history.sql        # Incremental history
│   └── schema.yml                  # Documentation
│
├── scripts/
│   ├── train.py                    # ⚡ Main training script (SGD)
│   ├── init_weights.py             # Xavier initialization
│   └── prepare_training_data.py    # Export train/val data
│
└── seeds/
    ├── input_images.csv            # Training images
    ├── true_labels.csv             # Training labels
    ├── validation_images.csv       # Validation images
    ├── validation_labels.csv       # Validation labels
    └── weights_*.csv, biases_*.csv # Model parameters
```

## How It Works

### Optimized Training Step

The key innovation is `training_step_sgd.sql` - a single SQL query that combines:

1. **Batch selection**: Filter to current mini-batch
2. **Forward pass**: Compute activations through all layers
3. **Backward pass**: Compute gradients via backpropagation
4. **Weight updates**: Apply SGD update rule

```sql
-- Single query does everything!
WITH 
  current_batch AS (SELECT ... WHERE batch_num = {{ batch_id }}),
  -- Forward pass
  l1_act AS (...), l2_act AS (...), softmax AS (...),
  -- Backward pass
  dz3 AS (...), grad_w3 AS (...), grad_w2 AS (...), grad_w1 AS (...),
  -- Weight updates
  fc1_new AS (SELECT w - lr * grad AS new_w FROM ...)
SELECT ... -- Updated weights + metrics
```

### Direct SQL Weight Updates

Instead of CSV → dbt seed → table (slow), we update weights directly:

```python
cursor.execute(f"""
    CREATE OR REPLACE TABLE {schema}.weights_fc1 AS
    SELECT output_idx, weights
    FROM {schema}.training_step_sgd
    WHERE param = 'weights_fc1'
""")
```

This eliminates the round-trip overhead between batches.

## Mini-batch SGD Benefits

| Benefit | Description |
|---------|-------------|
| **Faster convergence** | Multiple updates per epoch |
| **Escape local minima** | Gradient noise helps exploration |
| **Memory efficient** | Process subset of data at a time |
| **Regularization effect** | Noise acts as implicit regularization |

## Hyperparameter Guide

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `batch_size` | 16-32 | Smaller = more noise, larger = more stable |
| `learning_rate` | 0.01-0.1 | Start with 0.01, increase if too slow |
| `epochs` | 10-50 | More epochs for larger datasets |
| `--shuffle` | Yes | Helps prevent memorizing batch patterns |

## Troubleshooting

### "Missing seed files"
Run initialization scripts first:
```bash
python scripts/init_weights.py
python scripts/prepare_training_data.py
```

### Training is slow
- Reduce `--validate-every` (e.g., `--validate-every 5`)
- Use larger batch size
- Check Databricks cluster is running

### Loss not decreasing
- Try smaller learning rate
- Check data is normalized
- Verify weight initialization

---

**Status**: ✅ Ready for training  
**Method**: Mini-batch SGD (optimized)  
**Last Updated**: January 2026
