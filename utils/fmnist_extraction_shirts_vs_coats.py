#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os

# Use the datasets/our_data_fmnist directory
data_path = os.path.join(os.getcwd(), 'datasets', 'our_data_fmnist')

# Create output directory for extracted data
output_dir = os.path.join(os.getcwd(), 'extracted_datasets', 'extracted_data_fmnist')
os.makedirs(output_dir, exist_ok=True)

print("Reading Fashion-MNIST dataset from CSV files...")

# Read CSV files
train_path = os.path.join(data_path, "fashion-mnist_train.csv")
test_path = os.path.join(data_path, "fashion-mnist_test.csv")

try:
    print(f"Loading training data from {train_path}...")
    train = pd.read_csv(train_path)
    print(f"Loading test data from {test_path}...")
    test = pd.read_csv(test_path)
    
    print(f"Train set shape: {train.shape}")
    print(f"Test set shape: {test.shape}")
    print(f"Train labels distribution:\n{train['label'].value_counts().sort_index()}")
    print(f"Test labels distribution:\n{test['label'].value_counts().sort_index()}")
    
except FileNotFoundError as e:
    print(f"Error: Fashion-MNIST CSV files not found!")
    print(f"Details: {e}")
    print(f"\nExpected files in {data_path}:")
    print("  - fashion-mnist_train.csv")
    print("  - fashion-mnist_test.csv")
    exit(1)

# Extract function for Fashion-MNIST labels
def extract_fmnist(data, label, name_data, save=True):
    """
    Extract specific label from Fashion-MNIST dataset
    
    Args:
        data: DataFrame containing Fashion-MNIST data
        label: class label to extract (6 = Shirt, 8 = Coat)
        name_data: prefix for output filename ('train' or 'test')
        save: whether to save to CSV (default: True)
    """
    data_extracted = data[data['label'] == label]
    
    print(f"Found {len(data_extracted)} images of label {label} in {name_data} set")
    
    if save:
        # Save to output directory
        output_path = os.path.join(output_dir, f"{name_data}{label}.csv")
        data_extracted.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
        print(f"Shape: {data_extracted.shape}\n")
    
    return data_extracted

# Extract labels 6 (Shirt) and 8 (Coat) from both train and test sets
print("\n=== Extracting from Train Set ===")
print("Label 6 = Shirt")
train_6 = extract_fmnist(train, 6, 'train')
print("Label 8 = Coat")
train_8 = extract_fmnist(train, 8, 'train')

print("=== Extracting from Test Set ===")
print("Label 6 = Shirt")
test_6 = extract_fmnist(test, 6, 'test')
print("Label 8 = Coat")
test_8 = extract_fmnist(test, 8, 'test')

print("=" * 50)
print("Extraction complete!")
print(f"Output directory: {output_dir}")
print("Files created:")
print("  - train6.csv (Shirt images)")
print("  - train8.csv (Coat images)")
print("  - test6.csv (Shirt images)")
print("  - test8.csv (Coat images)")

