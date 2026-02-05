#!/usr/bin/env python3
"""
Initialize random weights for training from scratch.

Uses Xavier/Glorot initialization:
W ~ U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))

This creates the initial seed CSV files for the neural network.
"""

import numpy as np
import pandas as pd
import json
import os
from pathlib import Path

# Random seed for reproducibility
SEED = 42

# Network architecture
ARCHITECTURE = [
    (784, 128),   # fc1: 784 inputs -> 128 outputs
    (128, 64),    # fc2: 128 inputs -> 64 outputs
    (64, 10),     # fc3: 64 inputs -> 10 outputs
]

def xavier_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """Xavier/Glorot uniform initialization."""
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_out, fan_in))


def zero_init(size: int) -> np.ndarray:
    """Initialize biases to zero (standard practice)."""
    return np.zeros(size)


def save_weights_csv(weights: np.ndarray, filename: str, seeds_dir: Path):
    """Save weights as CSV with JSON arrays."""
    data = []
    for output_idx in range(weights.shape[0]):
        weight_row = weights[output_idx, :].tolist()
        data.append({
            'output_idx': output_idx,
            'weights': json.dumps(weight_row)
        })
    
    df = pd.DataFrame(data)
    filepath = seeds_dir / filename
    df.to_csv(filepath, index=False)
    print(f"  Saved {filename}: {weights.shape}")


def save_biases_csv(biases: np.ndarray, filename: str, seeds_dir: Path):
    """Save biases as CSV."""
    data = []
    for output_idx in range(biases.shape[0]):
        data.append({
            'output_idx': output_idx,
            'bias_value': biases[output_idx]
        })
    
    df = pd.DataFrame(data)
    filepath = seeds_dir / filename
    df.to_csv(filepath, index=False)
    print(f"  Saved {filename}: {biases.shape}")


def main():
    # Setup paths
    script_dir = Path(__file__).parent
    seeds_dir = script_dir.parent / 'seeds'
    seeds_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Initializing Neural Network Weights")
    print("=" * 60)
    print(f"Random seed: {SEED}")
    print(f"Output directory: {seeds_dir}")
    print()
    
    # Initialize random generator
    rng = np.random.default_rng(SEED)
    
    # Initialize each layer
    layer_names = ['fc1', 'fc2', 'fc3']
    
    for i, (fan_in, fan_out) in enumerate(ARCHITECTURE):
        layer_name = layer_names[i]
        print(f"Layer {i+1} ({layer_name}): {fan_in} -> {fan_out}")
        
        # Initialize weights
        weights = xavier_init(fan_in, fan_out, rng)
        save_weights_csv(weights, f'weights_{layer_name}.csv', seeds_dir)
        
        # Initialize biases (zeros)
        biases = zero_init(fan_out)
        save_biases_csv(biases, f'biases_{layer_name}.csv', seeds_dir)
        
        print()
    
    # Print summary statistics
    print("=" * 60)
    print("Initialization Summary")
    print("=" * 60)
    
    total_params = 0
    for fan_in, fan_out in ARCHITECTURE:
        total_params += fan_in * fan_out + fan_out
    
    print(f"Total parameters: {total_params:,}")
    print(f"  - fc1: {784 * 128 + 128:,} (weights: {784 * 128:,}, biases: 128)")
    print(f"  - fc2: {128 * 64 + 64:,} (weights: {128 * 64:,}, biases: 64)")
    print(f"  - fc3: {64 * 10 + 10:,} (weights: {64 * 10:,}, biases: 10)")
    print()
    print("Done! Run 'dbt seed' to load weights into Databricks.")


if __name__ == '__main__':
    main()

