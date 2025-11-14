#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import struct

# Use the datasets/our_data_mnist directory
data_path = os.path.join(os.getcwd(), 'datasets', 'our_data_mnist')

# Create output directory for extracted data
output_dir = os.path.join(os.getcwd(), 'extracted_datasets', 'extracted_data_mnist')
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

# Define file paths - check for both possible locations
train_images_path = os.path.join(data_path, "train-images.idx3-ubyte")
if not os.path.exists(train_images_path):
    train_images_path = os.path.join(data_path, "train-images-idx3-ubyte", "train-images-idx3-ubyte")

train_labels_path = os.path.join(data_path, "train-labels.idx1-ubyte")
if not os.path.exists(train_labels_path):
    train_labels_path = os.path.join(data_path, "train-labels-idx1-ubyte", "train-labels-idx1-ubyte")

test_images_path = os.path.join(data_path, "t10k-images.idx3-ubyte")
if not os.path.exists(test_images_path):
    test_images_path = os.path.join(data_path, "t10k-images-idx3-ubyte", "t10k-images-idx3-ubyte")

test_labels_path = os.path.join(data_path, "t10k-labels.idx1-ubyte")
if not os.path.exists(test_labels_path):
    test_labels_path = os.path.join(data_path, "t10k-labels-idx1-ubyte", "t10k-labels-idx1-ubyte")

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
    print(f"Error: MNIST IDX files not found!")
    print(f"Details: {e}")
    print(f"\nExpected files in {data_path}:")
    print("  - train-images.idx3-ubyte (or train-images-idx3-ubyte/train-images-idx3-ubyte)")
    print("  - train-labels.idx1-ubyte (or train-labels-idx1-ubyte/train-labels-idx1-ubyte)")
    print("  - t10k-images.idx3-ubyte (or t10k-images-idx3-ubyte/t10k-images-idx3-ubyte)")
    print("  - t10k-labels.idx1-ubyte (or t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte)")
    exit(1)

# Extract function for MNIST digits
def extract_mnist(data, label, name_data, n_samples=1000, save=True):
    """
    Extract specific label from MNIST dataset
    
    Args:
        data: DataFrame containing MNIST data
        label: digit label to extract (6 or 8)
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

# Extract 1000 images of digit 6 and digit 8 from both train and test sets
print("\n=== Extracting from Train Set ===")
train_6 = extract_mnist(train, 6, 'train', n_samples=1000)
train_8 = extract_mnist(train, 8, 'train', n_samples=1000)

print("=== Extracting from Test Set ===")
test_6 = extract_mnist(test, 6, 'test', n_samples=1000)
test_8 = extract_mnist(test, 8, 'test', n_samples=1000)

print("=" * 50)
print("Extraction complete!")
print(f"Output directory: {output_dir}")
print("Files created:")
print("  - train_6.csv (1000 images of digit 6)")
print("  - train_8.csv (1000 images of digit 8)")
print("  - test_6.csv (1000 images of digit 6)")
print("  - test_8.csv (1000 images of digit 8)")

