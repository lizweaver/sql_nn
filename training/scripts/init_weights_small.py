#!/usr/bin/env python3
"""
Initialize a SMALLER network for faster SQL training.

Architecture: 784 → 32 → 10 (only 2 layers, ~25K params vs 109K)
This is ~4x fewer parameters = ~4x faster training.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

SEED = 42
# Smaller architecture: just one hidden layer
ARCHITECTURE = [
    (784, 32),   # fc1: 784 -> 32
    (32, 10),    # fc2: 32 -> 10 (output)
]

def xavier_init(fan_in, fan_out, rng):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_out, fan_in))

def main():
    seeds_dir = Path(__file__).parent.parent / 'seeds'
    seeds_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Initializing SMALL Network (784 → 32 → 10)")
    print("=" * 60)
    
    rng = np.random.default_rng(SEED)
    
    total_params = 0
    for i, (fan_in, fan_out) in enumerate(ARCHITECTURE):
        layer = i + 1
        name = f'fc{layer}'
        
        # Weights as JSON (original format - works fine for small networks)
        weights = xavier_init(fan_in, fan_out, rng)
        weight_data = [
            {'output_idx': j, 'weights': json.dumps(weights[j].tolist())}
            for j in range(fan_out)
        ]
        pd.DataFrame(weight_data).to_csv(seeds_dir / f'weights_{name}.csv', index=False)
        
        # Biases
        biases = np.zeros(fan_out)
        bias_data = [{'output_idx': j, 'bias_value': biases[j]} for j in range(fan_out)]
        pd.DataFrame(bias_data).to_csv(seeds_dir / f'biases_{name}.csv', index=False)
        
        n_params = fan_in * fan_out + fan_out
        total_params += n_params
        print(f"  {name}: {fan_in} → {fan_out} ({n_params:,} params)")
    
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  (vs 109,386 in original = {109386/total_params:.1f}x smaller)")
    print("\nDone!")

if __name__ == '__main__':
    main()


