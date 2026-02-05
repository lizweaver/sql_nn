#!/usr/bin/env python3
"""
Initialize weights in PRE-EXPLODED format for fast training.

Instead of JSON arrays, stores weights as individual rows:
  (output_idx, input_idx, weight_value)

This eliminates expensive JSON parsing during training.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
ARCHITECTURE = [(784, 128), (128, 64), (64, 10)]

def xavier_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_out, fan_in))

def main():
    script_dir = Path(__file__).parent
    seeds_dir = script_dir.parent / 'seeds'
    seeds_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Initializing Weights (FAST/Pre-exploded Format)")
    print("=" * 60)
    
    rng = np.random.default_rng(SEED)
    layer_names = ['fc1', 'fc2', 'fc3']
    
    total_weight_rows = 0
    
    for i, (fan_in, fan_out) in enumerate(ARCHITECTURE):
        name = layer_names[i]
        print(f"\nLayer {name}: {fan_in} → {fan_out}")
        
        # Weights: pre-exploded format
        weights = xavier_init(fan_in, fan_out, rng)
        weight_rows = []
        for out_idx in range(fan_out):
            for in_idx in range(fan_in):
                weight_rows.append({
                    'output_idx': out_idx,
                    'input_idx': in_idx,
                    'weight_value': weights[out_idx, in_idx]
                })
        
        weights_df = pd.DataFrame(weight_rows)
        weights_df.to_csv(seeds_dir / f'weights_{name}_fast.csv', index=False)
        print(f"  weights_{name}_fast.csv: {len(weight_rows)} rows")
        total_weight_rows += len(weight_rows)
        
        # Biases: same format as before
        biases = np.zeros(fan_out)
        bias_df = pd.DataFrame([
            {'output_idx': j, 'bias_value': biases[j]} 
            for j in range(fan_out)
        ])
        bias_df.to_csv(seeds_dir / f'biases_{name}.csv', index=False)
        print(f"  biases_{name}.csv: {fan_out} rows")
    
    print(f"\n  Total weight rows: {total_weight_rows:,}")
    print("  (No JSON parsing needed during training!)")
    print("\nDone!")

if __name__ == '__main__':
    main()


