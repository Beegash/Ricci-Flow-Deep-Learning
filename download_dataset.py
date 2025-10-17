#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fashion MNIST veri setini indirip CSV formatına çevirir
"""
import os
import ssl
import numpy as np
import pandas as pd
from tensorflow import keras

# MacOS SSL sertifika sorununu çöz
ssl._create_default_https_context = ssl._create_unverified_context

def download_and_save_fmnist():
    """Fashion MNIST veri setini indirir ve CSV formatında kaydeder"""
    
    print("Fashion MNIST veri seti indiriliyor...")
    
    # Keras'tan Fashion MNIST veri setini indir
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    
    print(f"Train set: {x_train.shape}, Test set: {x_test.shape}")
    
    # Verileri düzleştir (28x28 -> 784)
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    
    # DataFrame'lere dönüştür
    train_df = pd.DataFrame(x_train_flat)
    train_df.insert(0, 'label', y_train)
    
    test_df = pd.DataFrame(x_test_flat)
    test_df.insert(0, 'label', y_test)
    
    # Kaydet
    print("Veri setleri CSV formatında kaydediliyor...")
    train_df.to_csv('fashion-mnist_train.csv', index=False)
    test_df.to_csv('fashion-mnist_test.csv', index=False)
    
    print("✓ fashion-mnist_train.csv kaydedildi")
    print("✓ fashion-mnist_test.csv kaydedildi")
    print(f"\nToplam train örnekleri: {len(train_df)}")
    print(f"Toplam test örnekleri: {len(test_df)}")
    
    return train_df, test_df

if __name__ == "__main__":
    download_and_save_fmnist()

