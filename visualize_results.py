#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ricci-NN Sonuçlarını Görselleştirme Script'i
Makale kalitesinde grafikler oluşturur
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

# Stil ayarları
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Çalışma dizinini ayarla
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def load_data():
    """Eğitim sonuçlarını yükle"""
    print("Veri yükleniyor...")
    
    model_predict = np.load("model_predict.npy", allow_pickle=True)
    accuracy = np.load("accuracy.npy")
    x_test = pd.read_csv("x_test.csv", header=None).values
    y_test = pd.read_csv("y_test.csv", header=None).values.flatten()
    
    return model_predict, accuracy, x_test, y_test

def plot_layer_activations_distribution(model_predict, save_path='layer_activations.png'):
    """Her katmandaki aktivasyon dağılımlarını görselleştir"""
    print("Katman aktivasyonları çiziliyor...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # İlk modeli kullan
    activations = model_predict[0]
    
    for i, (ax, layer_act) in enumerate(zip(axes, activations)):
        # Her katmanın aktivasyonlarını düzleştir
        flat_activations = layer_act.flatten()
        
        # Histogram
        ax.hist(flat_activations, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_title(f'Layer {i+1} Aktivasyon Dağılımı')
        ax.set_xlabel('Aktivasyon Değeri')
        ax.set_ylabel('Frekans')
        ax.grid(True, alpha=0.3)
        
        # İstatistikler ekle
        mean_val = np.mean(flat_activations)
        std_val = np.std(flat_activations)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Ortalama: {mean_val:.2f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Grafik kaydedildi: {save_path}")
    plt.close()

def plot_tsne_visualization(model_predict, y_test, save_path='tsne_layers.png'):
    """t-SNE ile katman temsillerini görselleştir"""
    print("t-SNE görselleştirmesi oluşturuluyor (bu biraz zaman alabilir)...")
    
    activations = model_predict[0]
    n_layers = len(activations)
    
    # 2x3 grid için en fazla 6 katman
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (ax, layer_act) in enumerate(zip(axes, activations)):
        # t-SNE uygula
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_result = tsne.fit_transform(layer_act)
        
        # Scatter plot
        scatter = ax.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                           c=y_test, cmap='coolwarm', alpha=0.6, s=20)
        ax.set_title(f'Layer {i+1} - t-SNE Projection')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        plt.colorbar(scatter, ax=ax, label='Class')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Grafik kaydedildi: {save_path}")
    plt.close()

def plot_pca_variance(model_predict, save_path='pca_variance.png'):
    """Her katmanda PCA ile varyans analizi"""
    print("PCA varyans analizi yapılıyor...")
    
    activations = model_predict[0]
    n_layers = len(activations)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (ax, layer_act) in enumerate(zip(axes, activations)):
        # PCA uygula
        n_components = min(50, layer_act.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(layer_act)
        
        # Kümülatif varyans
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        
        ax.plot(range(1, len(cumsum_var)+1), cumsum_var, 'b-', linewidth=2)
        ax.axhline(y=0.95, color='r', linestyle='--', label='95% Varyans')
        ax.set_title(f'Layer {i+1} - PCA Kümülatif Varyans')
        ax.set_xlabel('Bileşen Sayısı')
        ax.set_ylabel('Kümülatif Açıklanan Varyans')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Grafik kaydedildi: {save_path}")
    plt.close()

def plot_activation_heatmap(model_predict, save_path='activation_heatmap.png'):
    """Katmanlar arası aktivasyon korelasyon ısı haritası"""
    print("Aktivasyon korelasyon matrisi oluşturuluyor...")
    
    activations = model_predict[0]
    n_layers = len(activations)
    
    # Her katmanın ortalama aktivasyonunu hesapla
    mean_activations = []
    for layer_act in activations:
        mean_act = np.mean(layer_act, axis=0)
        mean_activations.append(mean_act)
    
    # Korelasyon matrisi
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Her katman için ilk 50 nöronu al (görselleştirme için)
    sample_size = min(50, min([len(m) for m in mean_activations]))
    correlation_data = np.array([m[:sample_size] for m in mean_activations])
    
    sns.heatmap(correlation_data, cmap='coolwarm', center=0, 
                xticklabels=False, yticklabels=[f'Layer {i+1}' for i in range(n_layers)],
                cbar_kws={'label': 'Ortalama Aktivasyon'}, ax=ax)
    ax.set_title('Katmanlar Arası Ortalama Aktivasyon Profili')
    ax.set_xlabel('Nöron İndeksi')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Grafik kaydedildi: {save_path}")
    plt.close()

def plot_model_comparison(accuracy, save_path='model_comparison.png'):
    """Farklı modellerin performans karşılaştırması"""
    print("Model karşılaştırması çiziliyor...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = [f'Model {i+1}' for i in range(len(accuracy))]
    colors = plt.cm.viridis(np.linspace(0, 1, len(accuracy)))
    
    bars = ax.bar(models, accuracy, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Değerleri çubukların üstüne yaz
    for i, (bar, acc) in enumerate(zip(bars, accuracy)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Test Accuracy', fontsize=14)
    ax.set_title('Model Performans Karşılaştırması', fontsize=16, fontweight='bold')
    ax.set_ylim([min(accuracy) - 0.01, max(accuracy) + 0.01])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Grafik kaydedildi: {save_path}")
    plt.close()

def create_all_visualizations():
    """Tüm görselleştirmeleri oluştur"""
    print("="*70)
    print("Ricci-NN Görselleştirme Script'i")
    print("="*70)
    
    # Veriyi yükle
    model_predict, accuracy, x_test, y_test = load_data()
    
    # Tüm grafikleri oluştur
    plot_model_comparison(accuracy)
    plot_layer_activations_distribution(model_predict)
    plot_activation_heatmap(model_predict)
    plot_pca_variance(model_predict)
    plot_tsne_visualization(model_predict, y_test)
    
    print("\n" + "="*70)
    print("✓ Tüm görselleştirmeler tamamlandı!")
    print("="*70)
    print("\nOluşturulan dosyalar:")
    print("  - model_comparison.png")
    print("  - layer_activations.png")
    print("  - activation_heatmap.png")
    print("  - pca_variance.png")
    print("  - tsne_layers.png")
    print("  - ricci_flow_analysis.png (knn.py'den)")

if __name__ == "__main__":
    create_all_visualizations()

