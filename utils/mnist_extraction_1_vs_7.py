#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import struct

# Use the mnist_dataset directory where you've placed the MNIST data
data_path = os.path.join(os.getcwd(), 'mnist_dataset')

# Create output directory for extracted data (in the parent directory)
output_dir = os.path.join(os.path.dirname(os.getcwd()), 'extracted_data_mnist')
os.makedirs(output_dir, exist_ok=True)

def read_idx_images(filename):
    """Read MNIST images from IDX file format"""
    with open(filename, 'rb') as f:
        # Read magic number and dimensions
        magic = struct.unpack('>I', f.read(4))[0]
        num_images = struct.unpack('>I', f.read(4))[0]
        rows = struct.unpack('>I', f.read(4))[0]
        cols = struct.unpack('>I', f.read(4))[0]
        
        # Read image data
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows * cols)
        
    return images

def read_idx_labels(filename):
    """Read MNIST labels from IDX file format"""
    with open(filename, 'rb') as f:
        # Read magic number and number of labels
        magic = struct.unpack('>I', f.read(4))[0]
        num_labels = struct.unpack('>I', f.read(4))[0]
        
        # Read label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        
    return labels

print("Reading MNIST dataset from IDX binary files...")

# Define file paths
train_images_path = os.path.join(data_path, "train-images.idx3-ubyte")
train_labels_path = os.path.join(data_path, "train-labels.idx1-ubyte")
test_images_path = os.path.join(data_path, "t10k-images.idx3-ubyte")
test_labels_path = os.path.join(data_path, "t10k-labels.idx1-ubyte")

try:
    # Read training data
    print("Loading training images...")
    train_images = read_idx_images(train_images_path)
    print("Loading training labels...")
    train_labels = read_idx_labels(train_labels_path)
    
    # Read test data
    print("Loading test images...")
    test_images = read_idx_images(test_images_path)
    print("Loading test labels...")
    test_labels = read_idx_labels(test_labels_path)
    
    # Create DataFrames (label as first column, then pixel columns)
    train = pd.DataFrame(train_images)
    train.insert(0, 'label', train_labels)
    
    test = pd.DataFrame(test_images)
    test.insert(0, 'label', test_labels)
    
    print(f"Train set shape: {train.shape}")
    print(f"Test set shape: {test.shape}")
    print(f"Train labels distribution: {np.bincount(train_labels)}")
    print(f"Test labels distribution: {np.bincount(test_labels)}")
    
except FileNotFoundError as e:
    print(f"Error: MNIST IDX files not found in {data_path}!")
    print(f"Details: {e}")
    print(f"\nExpected files:")
    print("  - train-images.idx3-ubyte")
    print("  - train-labels.idx1-ubyte")
    print("  - t10k-images.idx3-ubyte")
    print("  - t10k-labels.idx1-ubyte")
    exit(1)

# Extract function for MNIST digits 1 and 7
def extract_mnist(data, label, name_data, n_samples=1000, save=True):
    """
    Extract specific label from MNIST dataset
    
    Args:
        data: DataFrame containing MNIST data
        label: digit label to extract (1 or 7)
        name_data: prefix for output filename ('train' or 'test')
        n_samples: number of samples to extract (default: 1000)
        save: whether to save to CSV (default: True)
    """
    # Filter data by label
    data_label = data[data['label'] == label]
    
    print(f"Found {len(data_label)} images of digit {label} in {name_data} set")
    
    # Take only n_samples (1000) images
    if len(data_label) >= n_samples:
        data_extracted = data_label.head(n_samples)
        print(f"Extracting {n_samples} images of digit {label}")
    else:
        data_extracted = data_label
        print(f"Warning: Only {len(data_label)} images available (requested {n_samples})")
    
    if save:
        # Save to output directory
        output_path = os.path.join(output_dir, f"{name_data}_{label}.csv")
        data_extracted.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
        print(f"Shape: {data_extracted.shape}\n")
    
    return data_extracted

# Extract 1000 images of digit 1 and digit 7 from both train and test sets
print("\n=== Extracting from Train Set ===")
train_1 = extract_mnist(train, 1, 'train', n_samples=1000)
train_7 = extract_mnist(train, 7, 'train', n_samples=1000)

print("=== Extracting from Test Set ===")
test_1 = extract_mnist(test, 1, 'test', n_samples=1000)
test_7 = extract_mnist(test, 7, 'test', n_samples=1000)

print("=" * 50)
print("Extraction complete!")
print(f"Output directory: {output_dir}")
print("Files created:")
print("  - train_1.csv (1000 images of digit 1)")
print("  - train_7.csv (1000 images of digit 7)")
print("  - test_1.csv (1000 images of digit 1)")
print("  - test_7.csv (1000 images of digit 7)")

