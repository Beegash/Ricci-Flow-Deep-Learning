#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
# path = '/Users/anthonybaptista/Downloads/fMNIST_DNN_training/fmnist/'
# os.chdir(path)

# Dinamik yol - mevcut klasörde our_data_fmnist alt klasörünü kullan
data_path = os.path.join(os.getcwd(), 'our_data_fmnist')
os.chdir(data_path)

# test = pd.read_csv("fashion-mnist_test.csv")
# train = pd.read_csv("fashion-mnist_train.csv")

# Çıktı klasörü oluştur (ana klasörde extracted_data)
output_dir = os.path.join(os.path.dirname(os.getcwd()), 'extracted_data')
os.makedirs(output_dir, exist_ok=True)

test = pd.read_csv("fashion-mnist_test.csv")
train = pd.read_csv("fashion-mnist_train.csv")


# extract label 5 and 9

# def extract(data, label, name_data, save = True):
#     data_extracted = data[data['label'] == label]
#     if save == True:
#         data_extracted.to_csv(name_data + str(label) + ".csv", index = False)
    

# extract label 5 and 9
def extract(data, label, name_data, save = True):
    data_extracted = data[data['label'] == label]
    if save == True:
        # Çıktıyı extracted_data klasörüne yaz
        output_path = os.path.join(output_dir, name_data + str(label) + ".csv")
        data_extracted.to_csv(output_path, index = False)
        print(f"Saved: {output_path}")
        
extract(test, 5, 'test')
extract(test, 9, 'test')
extract(train, 5, 'train')
extract(train, 9, 'train')

