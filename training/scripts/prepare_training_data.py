#!/usr/bin/env python3
"""
Prepare Training Data

Exports MNIST training samples to CSV format for SQL training.
Creates separate train and validation splits.
"""

import torch
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def load_mnist_data(data_dir: Path, train: bool = True):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    dataset = datasets.MNIST(
        str(data_dir),
        train=train,
        download=True,
        transform=transform
    )
    
    return dataset


def export_samples(dataset, indices: list, seeds_dir: Path, 
                   image_filename: str, label_filename: str,
                   id_offset: int = 0):
    """Export samples to CSV files."""
    pixel_data = []
    label_data = []
    
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        sample_id = i + id_offset
        
        # Flatten image to 784 pixels
        flat_image = image.view(-1).numpy()
        
        # Add each pixel
        for pixel_idx, pixel_value in enumerate(flat_image):
            pixel_data.append({
                'sample_id': sample_id,
                'pixel_idx': pixel_idx,
                'pixel_value': float(pixel_value)
            })
        
        # Add label
        label_data.append({
            'sample_id': sample_id,
            'true_label': int(label)
        })
    
    # Save CSVs
    pixels_df = pd.DataFrame(pixel_data)
    pixels_path = seeds_dir / image_filename
    pixels_df.to_csv(pixels_path, index=False)
    print(f"  Saved {image_filename}: {len(indices)} samples, {len(pixel_data)} rows")
    
    labels_df = pd.DataFrame(label_data)
    labels_path = seeds_dir / label_filename
    labels_df.to_csv(labels_path, index=False)
    print(f"  Saved {label_filename}: {len(label_data)} rows")
    
    return label_data


def print_label_distribution(label_data: list, name: str):
    """Print the distribution of labels."""
    print(f"\n  {name} label distribution:")
    for label in range(10):
        count = sum(1 for l in label_data if l['true_label'] == label)
        bar = '█' * count
        print(f"    {label}: {count:3d} {bar}")


def main():
    parser = argparse.ArgumentParser(description='Prepare MNIST training data')
    parser.add_argument('--train-samples', type=int, default=100, 
                       help='Number of training samples')
    parser.add_argument('--val-samples', type=int, default=20,
                       help='Number of validation samples')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--stratified', action='store_true',
                       help='Use stratified sampling (equal per class)')
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    seeds_dir = project_dir / 'seeds'
    seeds_dir.mkdir(exist_ok=True)
    
    # Data directory (use shared data from initial_nn)
    data_dir = project_dir.parent / 'initial_nn' / 'data'
    
    print("=" * 60)
    print("Preparing Training & Validation Data")
    print("=" * 60)
    print(f"Training samples: {args.train_samples}")
    print(f"Validation samples: {args.val_samples}")
    print(f"Random seed: {args.seed}")
    print(f"Stratified: {args.stratified}")
    print(f"Output: {seeds_dir}")
    print()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Load dataset
    print("Loading MNIST dataset...")
    train_dataset = load_mnist_data(data_dir, train=True)
    test_dataset = load_mnist_data(data_dir, train=False)
    print(f"  Training set: {len(train_dataset)} samples")
    print(f"  Test set: {len(test_dataset)} samples")
    print()
    
    # Select indices
    if args.stratified:
        # Stratified sampling: equal samples per class
        print("Using stratified sampling...")
        train_per_class = args.train_samples // 10
        val_per_class = args.val_samples // 10
        
        # Get indices by class
        train_by_class = {i: [] for i in range(10)}
        val_by_class = {i: [] for i in range(10)}
        
        for idx in range(len(train_dataset)):
            _, label = train_dataset[idx]
            train_by_class[label].append(idx)
        
        for idx in range(len(test_dataset)):
            _, label = test_dataset[idx]
            val_by_class[label].append(idx)
        
        # Sample from each class
        train_indices = []
        val_indices = []
        
        for cls in range(10):
            np.random.shuffle(train_by_class[cls])
            np.random.shuffle(val_by_class[cls])
            train_indices.extend(train_by_class[cls][:train_per_class])
            val_indices.extend(val_by_class[cls][:val_per_class])
    else:
        # Random sampling
        print("Using random sampling...")
        all_train_indices = list(range(len(train_dataset)))
        all_test_indices = list(range(len(test_dataset)))
        
        np.random.shuffle(all_train_indices)
        np.random.shuffle(all_test_indices)
        
        train_indices = all_train_indices[:args.train_samples]
        val_indices = all_test_indices[:args.val_samples]
    
    # Export training data
    print(f"\nExporting training data ({len(train_indices)} samples)...")
    train_labels = export_samples(
        train_dataset, 
        train_indices, 
        seeds_dir,
        'input_images.csv',
        'true_labels.csv'
    )
    print_label_distribution(train_labels, "Training")
    
    # Export validation data
    print(f"\nExporting validation data ({len(val_indices)} samples)...")
    val_labels = export_samples(
        test_dataset,  # Use test set for validation
        val_indices, 
        seeds_dir,
        'validation_images.csv',
        'validation_labels.csv'
    )
    print_label_distribution(val_labels, "Validation")
    
    print("\n" + "=" * 60)
    print("Data Summary")
    print("=" * 60)
    print(f"  Training: {len(train_indices)} samples -> input_images.csv, true_labels.csv")
    print(f"  Validation: {len(val_indices)} samples -> validation_images.csv, validation_labels.csv")
    print()
    print("Run 'dbt seed' to load data into Databricks.")
    print("=" * 60)


if __name__ == '__main__':
    main()
