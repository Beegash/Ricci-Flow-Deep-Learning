#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spearman Correlation Analysis: Accuracy vs. Ricci Curvature (Rho)

Bu script, output_layers/ dizinindeki tüm model aktivasyonlarını analiz ederek:
1. Accuracy ve Rho (Ricci Curvature) değerlerini çıkarır
2. Spearman rank correlation hesaplar
3. Görselleştirmeler oluşturur
4. Dropout rate analizi yapar (eğer mevcut ise)
5. Parametrik testler uygular

Kullanım:
    python spearman_accuracy_rho_analysis.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, ttest_ind
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, triu as sp_triu
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import List
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# KONFIGÜRASYON
# ============================================================================

BASE_DIR = os.path.join(os.getcwd(), 'output_layers')
OUTPUT_DIR = os.path.join(os.getcwd(), 'spearman_analysis_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ACC_THRESHOLD: Sadece bu accuracy'nin üzerindeki modeller analiz edilir
ACC_THRESHOLD = 0.0  # Tüm modelleri dahil et (range geniş olsun)

# ============================================================================
# RICCI CURVATURE HESAPLAMA FONKSİYONLARI
# ============================================================================

def build_knn_graph(X: np.ndarray, k: int) -> csr_matrix:
    """kNN grafiği oluştur (undirected, unweighted)."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.dtype != np.float32 and X.dtype != np.float64:
        X = X.astype(np.float32, copy=False)
    
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(X)
    A = knn.kneighbors_graph(X, mode="connectivity")
    A = A.maximum(A.T)  # Symmetrize
    A.setdiag(0)
    A.eliminate_zeros()
    return A.tocsr()


def global_forman_ricci(A: csr_matrix) -> float:
    """Global Forman-Ricci curvature hesapla.
    
    Forman-Ricci curvature bir edge (i,j) için: R(i,j) = 4 - deg(i) - deg(j)
    Global Ricci (Rho) = tüm edge'lerin toplamı
    """
    deg = np.asarray(A.sum(axis=1)).ravel()
    A_ut = sp_triu(A, k=1).tocoo()
    curv = 4.0 - deg[A_ut.row] - deg[A_ut.col]
    return float(curv.sum())


def compute_rho_for_model(activations: List[np.ndarray], X0: np.ndarray, k: int) -> float:
    """Bir model için ortalama Ricci curvature (Rho) hesapla.
    
    Tüm layer'ların Ricci curvature'lerinin ortalamasını alır.
    Alternatif: Son layer'ın Ricci'sini veya toplamını da kullanabiliriz.
    """
    # Input layer için
    A0 = build_knn_graph(X0, k)
    ric0 = global_forman_ricci(A0)
    
    ric_list = [ric0]
    
    # Hidden layers için
    for Xl in activations:
        A = build_knn_graph(np.asarray(Xl), k)
        ric_l = global_forman_ricci(A)
        ric_list.append(ric_l)
    
    # Ortalama Rho (alternatif: toplam, son layer, vs.)
    rho = np.mean(ric_list)
    
    return rho


# ============================================================================
# VERİ ÇIKARIMI
# ============================================================================

def extract_run_data(combo_dir: str) -> list:
    """Bir architecture+dataset kombinasyonundan tüm run'ları çıkar.
    
    Returns:
        List of dicts, her dict bir run'ı temsil eder:
        {
            'architecture': str,
            'dataset': str,
            'k': int,
            'model_idx': int,
            'accuracy': float,
            'rho': float,
            'dropout_rate': float or None,
            'run_id': str
        }
    """
    runs = []
    
    # Combo dizininden architecture ve dataset ismini çıkar
    combo_name = os.path.basename(combo_dir)
    parts = combo_name.split('_')
    
    # Architecture ve dataset'i ayır (örn: "bottleneck_5_fmnist_sandals_vs_boots")
    if len(parts) >= 3:
        # Architecture kısmını bul (ilk 2 veya 3 parça)
        arch_parts = []
        dataset_parts = []
        found_dataset = False
        
        for part in parts:
            if part in ['mnist', 'fmnist', 'synthetic'] and not found_dataset:
                found_dataset = True
                dataset_parts.append(part)
            elif found_dataset:
                dataset_parts.append(part)
            else:
                arch_parts.append(part)
        
        architecture = '_'.join(arch_parts) if arch_parts else combo_name
        dataset = '_'.join(dataset_parts) if dataset_parts else combo_name
    else:
        architecture = combo_name
        dataset = combo_name
    
    # models_b70 klasörünü kontrol et
    models_dir = os.path.join(combo_dir, 'models_b70')
    if not os.path.exists(models_dir):
        return runs
    
    # Model dosyalarını yükle
    model_predict_path = os.path.join(models_dir, 'model_predict.npy')
    accuracy_path = os.path.join(models_dir, 'accuracy.npy')
    x_test_path = os.path.join(models_dir, 'x_test.csv')
    
    if not all(os.path.exists(p) for p in [model_predict_path, accuracy_path, x_test_path]):
        return runs
    
    try:
        model_predict = np.load(model_predict_path, allow_pickle=True)
        accuracy = np.load(accuracy_path)
        X0 = pd.read_csv(x_test_path, header=None).values
        
        # Tüm analysis_k* klasörlerini bul
        analysis_dirs = [d for d in os.listdir(combo_dir) 
                        if d.startswith('analysis_k') and os.path.isdir(os.path.join(combo_dir, d))]
        
        for analysis_dir in analysis_dirs:
            # k değerini çıkar
            try:
                k = int(analysis_dir.split('_k')[1])
            except:
                continue
            
            # Her model için rho hesapla
            for model_idx in range(len(model_predict)):
                if model_idx >= len(accuracy):
                    continue
                
                acc = float(accuracy[model_idx])
                
                # Accuracy threshold kontrolü
                if acc < ACC_THRESHOLD:
                    continue
                
                # Rho hesapla
                try:
                    activations = model_predict[model_idx]
                    rho = compute_rho_for_model(activations, X0, k)
                    
                    run_id = f"{combo_name}_k{k}_m{model_idx}"
                    
                    runs.append({
                        'architecture': architecture,
                        'dataset': dataset,
                        'k': k,
                        'model_idx': model_idx,
                        'accuracy': acc,
                        'rho': rho,
                        'dropout_rate': None,  # Dropout bilgisi metadata'da yoksa None
                        'run_id': run_id
                    })
                except Exception as e:
                    print(f"  [WARN] Error computing rho for {run_id}: {e}")
                    continue
                    
    except Exception as e:
        print(f"  [WARN] Error loading {combo_dir}: {e}")
    
    return runs


def collect_all_runs() -> pd.DataFrame:
    """Tüm output_layers/ dizininden run'ları topla."""
    print("=" * 80)
    print("VERİ ÇIKARIMI: output_layers/ dizininden run'lar toplanıyor...")
    print("=" * 80)
    
    all_runs = []
    
    # Tüm combo dizinlerini bul
    combo_dirs = [os.path.join(BASE_DIR, d) 
                  for d in os.listdir(BASE_DIR) 
                  if os.path.isdir(os.path.join(BASE_DIR, d)) and not d.endswith('.csv')]
    
    print(f"Toplam {len(combo_dirs)} kombinasyon bulundu.")
    
    for combo_dir in tqdm(combo_dirs, desc="Kombinasyonlar işleniyor"):
        runs = extract_run_data(combo_dir)
        all_runs.extend(runs)
    
    df = pd.DataFrame(all_runs)
    
    print(f"\n✓ Toplam {len(df)} run toplandı.")
    print(f"  - Unique architectures: {df['architecture'].nunique()}")
    print(f"  - Unique datasets: {df['dataset'].nunique()}")
    print(f"  - Unique k values: {df['k'].nunique()}")
    print(f"  - Accuracy range: [{df['accuracy'].min():.4f}, {df['accuracy'].max():.4f}]")
    print(f"  - Rho range: [{df['rho'].min():.2e}, {df['rho'].max():.2e}]")
    
    return df


# ============================================================================
# SPEARMAN KORELASYON ANALİZİ
# ============================================================================

def compute_spearman_correlation(df: pd.DataFrame) -> dict:
    """Accuracy ve Rho arasındaki Spearman korelasyonunu hesapla."""
    print("\n" + "=" * 80)
    print("SPEARMAN RANK CORRELATION ANALİZİ")
    print("=" * 80)
    
    # Accuracy'ye göre sıralama
    df_sorted_acc = df.sort_values('accuracy', ascending=False).reset_index(drop=True)
    df_sorted_acc['rank_accuracy'] = df_sorted_acc.index + 1
    
    # Rho'ya göre sıralama
    df_sorted_rho = df.sort_values('rho', ascending=False).reset_index(drop=True)
    df_sorted_rho['rank_rho'] = df_sorted_rho.index + 1
    
    # Merge
    df_merged = df_sorted_acc.merge(
        df_sorted_rho[['run_id', 'rank_rho']], 
        on='run_id', 
        how='inner'
    )
    
    # Spearman korelasyonu hesapla
    spearman_corr, spearman_p = spearmanr(
        df_merged['rank_accuracy'].values,
        df_merged['rank_rho'].values
    )
    
    # Pearson korelasyonu da hesapla (karşılaştırma için)
    pearson_corr, pearson_p = pearsonr(
        df_merged['accuracy'].values,
        df_merged['rho'].values
    )
    
    results = {
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_p,
        'pearson_correlation': pearson_corr,
        'pearson_p_value': pearson_p,
        'n_samples': len(df_merged)
    }
    
    print(f"\n📊 SPEARMAN KORELASYON SONUÇLARI:")
    print(f"   Spearman ρ (rank correlation): {spearman_corr:.4f}")
    print(f"   p-value: {spearman_p:.2e}")
    print(f"   N (sample size): {results['n_samples']}")
    print(f"\n📈 PEARSON KORELASYON (karşılaştırma):")
    print(f"   Pearson r: {pearson_corr:.4f}")
    print(f"   p-value: {pearson_p:.2e}")
    
    if spearman_corr > 0.7:
        print(f"\n✓ Güçlü pozitif korelasyon! (ρ > 0.7)")
    elif spearman_corr > 0.5:
        print(f"\n✓ Orta pozitif korelasyon (0.5 < ρ < 0.7)")
    elif spearman_corr > 0.3:
        print(f"\n⚠ Zayıf pozitif korelasyon (0.3 < ρ < 0.5)")
    else:
        print(f"\n⚠ Çok zayıf veya negatif korelasyon (ρ < 0.3)")
    
    return results, df_merged


# ============================================================================
# GÖRSELLEŞTİRME
# ============================================================================

def create_visualizations(df: pd.DataFrame, spearman_results: dict, df_merged: pd.DataFrame):
    """Tüm görselleştirmeleri oluştur."""
    print("\n" + "=" * 80)
    print("GÖRSELLEŞTİRME OLUŞTURULUYOR...")
    print("=" * 80)
    
    # Style
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Scatter Plot: Accuracy vs. Rho
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1a. Ana scatter plot
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df['accuracy'], df['rho'], 
                         alpha=0.6, s=50, c=df['k'], 
                         cmap='viridis', edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Ricci Curvature (Rho)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Accuracy vs. Ricci Curvature\n(Spearman ρ = {spearman_results["spearman_correlation"]:.4f}, p = {spearman_results["spearman_p_value"]:.2e})', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='k value')
    
    # Trend line ekle
    z = np.polyfit(df['accuracy'], df['rho'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['accuracy'].min(), df['accuracy'].max(), 100)
    ax1.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Linear fit')
    ax1.legend()
    
    # 1b. Log scale (Rho için)
    ax2 = axes[0, 1]
    # Rho negatif olabilir, mutlak değer al
    rho_abs = np.abs(df['rho'])
    scatter2 = ax2.scatter(df['accuracy'], rho_abs, 
                          alpha=0.6, s=50, c=df['k'], 
                          cmap='viridis', edgecolors='black', linewidth=0.5)
    ax2.set_yscale('log')
    ax2.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_ylabel('|Ricci Curvature| (log scale)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy vs. |Rho| (Log Scale)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='k value')
    
    # 1c. Rank scatter plot
    ax3 = axes[1, 0]
    ax3.scatter(df_merged['rank_accuracy'], df_merged['rank_rho'], 
               alpha=0.6, s=50, c=df_merged['k'], 
               cmap='viridis', edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Rank by Accuracy', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Rank by Rho', fontsize=12, fontweight='bold')
    ax3.set_title('Rank Comparison (Spearman Correlation)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Diagonal line (perfect correlation)
    max_rank = max(df_merged['rank_accuracy'].max(), df_merged['rank_rho'].max())
    ax3.plot([1, max_rank], [1, max_rank], 'r--', alpha=0.5, linewidth=2, label='Perfect correlation')
    ax3.legend()
    
    # 1d. Distribution plots
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    
    ax4.hist(df['accuracy'], bins=50, alpha=0.6, color='steelblue', label='Accuracy')
    ax4_twin.hist(df['rho'], bins=50, alpha=0.6, color='coral', label='Rho')
    
    ax4.set_xlabel('Value', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency (Accuracy)', fontsize=12, fontweight='bold', color='steelblue')
    ax4_twin.set_ylabel('Frequency (Rho)', fontsize=12, fontweight='bold', color='coral')
    ax4.set_title('Distribution of Accuracy and Rho', fontsize=14, fontweight='bold')
    ax4.tick_params(axis='y', labelcolor='steelblue')
    ax4_twin.tick_params(axis='y', labelcolor='coral')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_vs_rho_scatter.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Scatter plot kaydedildi: accuracy_vs_rho_scatter.png")
    plt.close()
    
    # 2. Architecture ve Dataset bazlı analizler
    if df['architecture'].nunique() > 1:
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # 2a. Architecture bazlı
        ax1 = axes[0, 0]
        for arch in df['architecture'].unique():
            arch_data = df[df['architecture'] == arch]
            ax1.scatter(arch_data['accuracy'], arch_data['rho'], 
                       alpha=0.6, s=50, label=arch)
        ax1.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Rho', fontsize=12, fontweight='bold')
        ax1.set_title('Accuracy vs. Rho by Architecture', fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2b. Dataset bazlı
        ax2 = axes[0, 1]
        for dataset in df['dataset'].unique():
            dataset_data = df[df['dataset'] == dataset]
            ax2.scatter(dataset_data['accuracy'], dataset_data['rho'], 
                       alpha=0.6, s=50, label=dataset)
        ax2.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Rho', fontsize=12, fontweight='bold')
        ax2.set_title('Accuracy vs. Rho by Dataset', fontsize=14, fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 2c. k value bazlı
        ax3 = axes[1, 0]
        for k_val in sorted(df['k'].unique()):
            k_data = df[df['k'] == k_val]
            ax3.scatter(k_data['accuracy'], k_data['rho'], 
                       alpha=0.6, s=50, label=f'k={k_val}')
        ax3.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Rho', fontsize=12, fontweight='bold')
        ax3.set_title('Accuracy vs. Rho by k value', fontsize=14, fontweight='bold')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 2d. Box plots
        ax4 = axes[1, 1]
        df_melted = df.melt(id_vars=['architecture'], 
                           value_vars=['accuracy', 'rho'], 
                           var_name='metric', value_name='value')
        sns.boxplot(data=df_melted, x='architecture', y='value', hue='metric', ax=ax4)
        ax4.set_title('Distribution by Architecture', fontsize=14, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'grouped_analysis.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Grouped analysis kaydedildi: grouped_analysis.png")
        plt.close()
    
    # 3. Dropout rate analizi (eğer mevcut ise)
    if df['dropout_rate'].notna().any():
        create_dropout_analysis(df)
    else:
        print("⚠ Dropout rate bilgisi mevcut değil, dropout analizi atlandı.")


def create_dropout_analysis(df: pd.DataFrame):
    """Dropout rate analizi görselleştirmesi."""
    df_dropout = df[df['dropout_rate'].notna()]
    
    if len(df_dropout) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Dropout rate'a göre grupla
    for ax_idx, (dropout_val, group_data) in enumerate(df_dropout.groupby('dropout_rate')):
        if ax_idx >= 4:
            break
        row, col = ax_idx // 2, ax_idx % 2
        ax = axes[row, col]
        
        ax.scatter(group_data['accuracy'], group_data['rho'], 
                  alpha=0.6, s=50, label=f'Dropout={dropout_val}')
        ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_ylabel('Rho', fontsize=12, fontweight='bold')
        ax.set_title(f'Accuracy vs. Rho (Dropout={dropout_val})', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dropout_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Dropout analysis kaydedildi: dropout_analysis.png")
    plt.close()


# ============================================================================
# PARAMETRİK TESTLER
# ============================================================================

def parametric_tests(df: pd.DataFrame):
    """Parametrik testler: 'Daha iyi network performansı = Daha iyi Ricci skoru' hipotezini test et."""
    print("\n" + "=" * 80)
    print("PARAMETRİK TESTLER")
    print("=" * 80)
    
    # Accuracy'yi yüksek/düşük olarak iki gruba ayır
    median_acc = df['accuracy'].median()
    high_acc = df[df['accuracy'] >= median_acc]
    low_acc = df[df['accuracy'] < median_acc]
    
    print(f"\n📊 Grup İstatistikleri:")
    print(f"   Yüksek Accuracy (≥ {median_acc:.4f}): N = {len(high_acc)}")
    print(f"   Düşük Accuracy (< {median_acc:.4f}): N = {len(low_acc)}")
    print(f"   Yüksek Accuracy - Ortalama Rho: {high_acc['rho'].mean():.2e} ± {high_acc['rho'].std():.2e}")
    print(f"   Düşük Accuracy - Ortalama Rho: {low_acc['rho'].mean():.2e} ± {low_acc['rho'].std():.2e}")
    
    # T-test
    t_stat, t_p = ttest_ind(high_acc['rho'], low_acc['rho'])
    
    print(f"\n🔬 T-TEST SONUÇLARI:")
    print(f"   T-statistic: {t_stat:.4f}")
    print(f"   p-value: {t_p:.2e}")
    
    if t_p < 0.001:
        print(f"   ✓ Çok yüksek anlamlılık (p < 0.001)")
    elif t_p < 0.01:
        print(f"   ✓ Yüksek anlamlılık (p < 0.01)")
    elif t_p < 0.05:
        print(f"   ✓ Anlamlı (p < 0.05)")
    else:
        print(f"   ⚠ Anlamlı değil (p ≥ 0.05)")
    
    # Görselleştirme
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box plot
    ax1 = axes[0]
    data_for_box = [low_acc['rho'], high_acc['rho']]
    bp = ax1.boxplot(data_for_box, labels=[f'Low Acc\n(N={len(low_acc)})', 
                                           f'High Acc\n(N={len(high_acc)})'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightblue')
    ax1.set_ylabel('Ricci Curvature (Rho)', fontsize=12, fontweight='bold')
    ax1.set_title(f'T-Test: High vs. Low Accuracy\n(p = {t_p:.2e})', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Violin plot
    ax2 = axes[1]
    df_test = pd.DataFrame({
        'Rho': list(low_acc['rho']) + list(high_acc['rho']),
        'Group': ['Low Accuracy'] * len(low_acc) + ['High Accuracy'] * len(high_acc)
    })
    sns.violinplot(data=df_test, x='Group', y='Rho', ax=ax2, palette=['lightcoral', 'lightblue'])
    ax2.set_ylabel('Ricci Curvature (Rho)', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'parametric_tests.png'), dpi=300, bbox_inches='tight')
    print(f"\n✓ Parametrik test görselleştirmesi kaydedildi: parametric_tests.png")
    plt.close()
    
    return {
        't_statistic': t_stat,
        't_p_value': t_p,
        'high_acc_mean_rho': high_acc['rho'].mean(),
        'low_acc_mean_rho': low_acc['rho'].mean(),
        'high_acc_std_rho': high_acc['rho'].std(),
        'low_acc_std_rho': low_acc['rho'].std()
    }


# ============================================================================
# ANA FONKSİYON
# ============================================================================

def main():
    """Ana analiz pipeline'ı."""
    print("\n" + "=" * 80)
    print("SPEARMAN CORRELATION ANALYSIS: ACCURACY vs. RICCI CURVATURE (RHO)")
    print("=" * 80)
    
    # 1. Veri çıkarımı
    df = collect_all_runs()
    
    if len(df) == 0:
        print("\n❌ HATA: Hiç run bulunamadı!")
        return
    
    # CSV olarak kaydet
    df.to_csv(os.path.join(OUTPUT_DIR, 'all_runs_data.csv'), index=False)
    print(f"\n✓ Tüm run verileri kaydedildi: all_runs_data.csv")
    
    # 2. Spearman korelasyon analizi
    spearman_results, df_merged = compute_spearman_correlation(df)
    
    # 3. Görselleştirmeler
    create_visualizations(df, spearman_results, df_merged)
    
    # 4. Parametrik testler
    parametric_results = parametric_tests(df)
    
    # 5. Özet rapor
    print("\n" + "=" * 80)
    print("ÖZET RAPOR")
    print("=" * 80)
    
    summary = {
        'total_runs': len(df),
        'spearman_correlation': spearman_results['spearman_correlation'],
        'spearman_p_value': spearman_results['spearman_p_value'],
        'pearson_correlation': spearman_results['pearson_correlation'],
        'pearson_p_value': spearman_results['pearson_p_value'],
        't_test_p_value': parametric_results['t_p_value'],
        'high_acc_mean_rho': parametric_results['high_acc_mean_rho'],
        'low_acc_mean_rho': parametric_results['low_acc_mean_rho']
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'summary_report.csv'), index=False)
    
    print(f"\n📋 ÖZET:")
    print(f"   Toplam Run Sayısı: {summary['total_runs']}")
    print(f"   Spearman ρ: {summary['spearman_correlation']:.4f} (p = {summary['spearman_p_value']:.2e})")
    print(f"   Pearson r: {summary['pearson_correlation']:.4f} (p = {summary['pearson_p_value']:.2e})")
    print(f"   T-Test p-value: {summary['t_test_p_value']:.2e}")
    print(f"\n✓ Özet rapor kaydedildi: summary_report.csv")
    print(f"\n✓ Tüm sonuçlar '{OUTPUT_DIR}' klasörüne kaydedildi.")
    print("=" * 80)


if __name__ == "__main__":
    main()

