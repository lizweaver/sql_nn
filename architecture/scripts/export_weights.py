"""
Export PyTorch model weights to CSV files for dbt seeds.

This script creates Databricks format:
- Weights: (output_idx, weights) as JSON array string
- Biases: (output_idx, bias_value)
- Input images: (sample_id, pixel_idx, pixel_value)
- True labels: (sample_id, true_label)

Usage:
    python export_weights.py --model_path ../initial_nn/model.pth --output_dir ../seeds
"""

import torch
import numpy as np
import pandas as pd
import json
import argparse
import os
import sys

# Add initial_nn to path so we can import the model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'initial_nn'))
from model import Net


def export_array_format_databricks(weights, layer_name, output_dir):
    """
    Export weights in array format for Databricks: (output_idx, weights)
    The weights column contains a JSON array of weight values.
    """
    out_dim = weights.shape[0]
    
    # Convert each row to JSON array format (same as Snowflake)
    weights_json = [json.dumps(row.tolist()) for row in weights]
    
    df = pd.DataFrame({
        'output_idx': np.arange(out_dim),
        'weights': weights_json
    })
    
    # Use plural 'weights' for array format filenames
    filename = f'{layer_name}s_databricks.csv' if 'weight' in layer_name else f'{layer_name}_databricks.csv'
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"  Exported {layer_name} Databricks format: {len(df)} rows -> {filepath}")
    return df


def export_bias(biases, layer_name, output_dir):
    """
    Export biases: (output_idx, bias_value)
    Same format for all implementations.
    """
    df = pd.DataFrame({
        'output_idx': np.arange(len(biases)),
        'bias_value': biases
    })
    filepath = os.path.join(output_dir, f'{layer_name}.csv')
    df.to_csv(filepath, index=False)
    print(f"  Exported {layer_name}: {len(df)} rows -> {filepath}")
    return df


def create_sample_input(output_dir, num_samples=10):
    """
    Create a sample input CSV from MNIST test data.
    Format: (sample_id, pixel_idx, pixel_value)
    """
    try:
        from torchvision import datasets, transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        test_dataset = datasets.MNIST(
            os.path.join(os.path.dirname(__file__), '..', '..', 'initial_nn', 'data'),
            train=False,
            download=False,
            transform=transform
        )
        
        rows = []
        labels = []
        
        for sample_id in range(min(num_samples, len(test_dataset))):
            image, label = test_dataset[sample_id]
            image = image.view(-1).numpy()  # Flatten to 784 pixels
            
            labels.append({
                'sample_id': sample_id,
                'true_label': label
            })
            
            for pixel_idx in range(len(image)):
                rows.append({
                    'sample_id': sample_id,
                    'pixel_idx': pixel_idx,
                    'pixel_value': float(image[pixel_idx])
                })
        
        # Save input images
        df_images = pd.DataFrame(rows)
        filepath_images = os.path.join(output_dir, 'input_images.csv')
        df_images.to_csv(filepath_images, index=False)
        print(f"  Exported input_images: {len(rows)} rows ({num_samples} samples) -> {filepath_images}")
        
        # Save true labels for validation
        df_labels = pd.DataFrame(labels)
        filepath_labels = os.path.join(output_dir, 'true_labels.csv')
        df_labels.to_csv(filepath_labels, index=False)
        print(f"  Exported true_labels: {len(labels)} rows -> {filepath_labels}")
        
        return df_images, df_labels
        
    except Exception as e:
        print(f"  Warning: Could not create sample input: {e}")
        print("  You'll need to create input_images.csv manually")
        return None, None


def main(model_path, output_dir, num_samples=10):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = Net()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    print("\nExporting weights and biases...")
    
    # Export each layer
    for name, param in model.named_parameters():
        data = param.detach().numpy()
        layer_name = name.replace('.', '_')  # fc1.weight -> fc1_weight
        
        if 'weight' in name:
            print(f"\n{name}: shape {data.shape}")
            # Export Databricks format
            export_array_format_databricks(data, layer_name, output_dir)
        elif 'bias' in name:
            print(f"\n{name}: shape {data.shape}")
            export_bias(data, layer_name, output_dir)
    
    # Create sample input data
    print("\nCreating sample input data...")
    create_sample_input(output_dir, num_samples)
    
    print("\n✓ Export complete!")
    print(f"\nSeeds created in: {output_dir}")
    print("\nTo load into dbt, run:")
    print("  dbt seed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export PyTorch weights to dbt seeds')
    parser.add_argument('--model_path', type=str, 
                        default='../../initial_nn/model.pth',
                        help='Path to the PyTorch model file')
    parser.add_argument('--output_dir', type=str,
                        default='../seeds',
                        help='Output directory for CSV seeds')
    parser.add_argument('--num_samples', type=int,
                        default=10,
                        help='Number of sample images to export')
    
    args = parser.parse_args()
    
    # Resolve paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, args.model_path)
    output_dir = os.path.join(script_dir, args.output_dir)
    
    main(model_path, output_dir, args.num_samples)

