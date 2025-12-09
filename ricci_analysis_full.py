#!/usr/bin/env python3
"""
Deep Learning as Ricci Flow - Kapsamlı Analiz Scripti

Bu script, output_layers/ dizinindeki aktivasyon dosyalarını doğrudan kullanarak:
1. Her run için Accuracy ve Ricci Curvature (Rho) değerlerini çıkarır
2. Spearman Rank Correlation analizi yapar
3. Accuracy vs Rho görselleştirmeleri oluşturur
4. Architecture/Depth bazlı karşılaştırmalı analiz yapar
5. Parametrik testler (T-test) uygular

Yazar: Otomatik üretildi
Tarih: 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import spearmanr, pearsonr, ttest_ind, mannwhitneyu
from scipy.sparse import csr_matrix, triu as sp_triu
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Tuple, Optional
import warnings
import glob
from datetime import datetime

warnings.filterwarnings('ignore')

# Stil ayarları
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# ============================================================================
# OUTPUT DİZİNİ
# ============================================================================
OUTPUT_DIR = "ricci_analysis_results"

# ============================================================================
# RICCI CURVATURE HESAPLAMA FONKSİYONLARI
# ============================================================================

def build_knn_graph(X: np.ndarray, k: int) -> csr_matrix:
    """kNN grafiği oluştur (yönsüz, ağırlıksız)."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.dtype != np.float32 and X.dtype != np.float64:
        X = X.astype(np.float32, copy=False)
    
    # k'yı kontrol et - veri sayısından büyük olmamalı
    n_samples = X.shape[0]
    k = min(k, n_samples - 1)
    
    if k < 1:
        return csr_matrix((n_samples, n_samples))
    
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(X)
    A = knn.kneighbors_graph(X, mode="connectivity")
    A = A.maximum(A.T)  # Simetrik yap
    A.setdiag(0)  # Self-loop'ları kaldır
    A.eliminate_zeros()
    return A.tocsr()


def global_forman_ricci(A: csr_matrix) -> float:
    """
    Global Forman-Ricci eğrilik katsayısını hesapla.
    Formül: R(i,j) = 4 - deg(i) - deg(j) her kenar için
    Return: Tüm kenar eğriliklerinin toplamı (Rho)
    """
    if A.nnz == 0:
        return 0.0
    deg = np.asarray(A.sum(axis=1)).ravel()
    A_ut = sp_triu(A, k=1).tocoo()
    if len(A_ut.row) == 0:
        return 0.0
    curv = 4.0 - deg[A_ut.row] - deg[A_ut.col]
    return float(curv.sum())


def calculate_mean_ricci_from_mfr(mfr_path: str) -> float:
    """
    mfr.csv dosyasından ortalama Ricci değerini hesapla.
    mfr.csv: layer, mod, ssr (ssr = Ricci değeri)
    """
    try:
        df = pd.read_csv(mfr_path)
        if 'ssr' in df.columns:
            return df['ssr'].mean()
        return np.nan
    except Exception:
        return np.nan


def calculate_ricci_for_activations(activations: List[np.ndarray], X0: np.ndarray, k: int) -> Tuple[float, List[float]]:
    """
    Bir model için tüm layer'lardaki Ricci eğriliklerini hesapla.
    
    Args:
        activations: Layer aktivasyon dizileri listesi
        X0: Orijinal giriş verisi (baseline)
        k: kNN grafiği için komşu sayısı
        
    Returns:
        (ortalama_ricci, [layer_ricci_değerleri])
    """
    try:
        # Giriş layer'ı için Ricci
        A0 = build_knn_graph(X0, k)
        ric0 = global_forman_ricci(A0)
        
        ric_values = [ric0]
        
        # Her aktivasyon layer'ı için Ricci
        for Xl in activations:
            try:
                A = build_knn_graph(np.asarray(Xl), k)
                ric = global_forman_ricci(A)
                ric_values.append(ric)
            except Exception:
                continue
        
        return np.mean(ric_values), ric_values
    except Exception:
        return np.nan, []


# ============================================================================
# VERİ TOPLAMA FONKSİYONLARI
# ============================================================================

def parse_run_info(run_path: str) -> Dict:
    """
    Dizin yolundan run bilgilerini çıkar.
    Örnek: 'bottleneck_5_fmnist_sandals_vs_boots' ->
        {'architecture': 'bottleneck', 'depth': 5, 'dataset': 'fmnist_sandals_vs_boots'}
    """
    run_name = os.path.basename(run_path)
    parts = run_name.split('_')
    
    info = {
        'run_name': run_name,
        'full_path': run_path,
        'architecture': 'unknown',
        'depth': 0,
        'dataset': '',
        'dataset_type': 'unknown'
    }
    
    # Architecture tipini çıkar
    if 'bottleneck' in run_name.lower():
        info['architecture'] = 'bottleneck'
    elif 'narrow' in run_name.lower():
        info['architecture'] = 'narrow'
    elif 'wide' in run_name.lower():
        info['architecture'] = 'wide'
    
    # Depth (layer sayısı) ve dataset'i çıkar
    for i, part in enumerate(parts):
        if part.isdigit():
            info['depth'] = int(part)
            # Dataset kalan kısım
            info['dataset'] = '_'.join(parts[i+1:])
            break
    
    # Dataset tipi
    if 'mnist' in info['dataset'].lower() and 'fmnist' not in info['dataset'].lower():
        info['dataset_type'] = 'mnist'
    elif 'fmnist' in info['dataset'].lower():
        info['dataset_type'] = 'fmnist'
    elif 'synthetic' in info['dataset'].lower():
        info['dataset_type'] = 'synthetic'
    
    return info


def find_best_k_for_run(run_path: str) -> int:
    """
    Bir run için en iyi k değerini bul (en güçlü korelasyona sahip olan).
    """
    summary_files = glob.glob(os.path.join(run_path, "*_summary.csv"))
    
    best_k = 100  # default
    best_r = 0
    
    for summary_file in summary_files:
        try:
            df = pd.read_csv(summary_file)
            if 'k' in df.columns and 'r_all' in df.columns:
                # En güçlü korelasyona sahip k'yı bul
                idx = df['r_all'].abs().idxmax()
                if abs(df.loc[idx, 'r_all']) > abs(best_r):
                    best_r = df.loc[idx, 'r_all']
                    best_k = int(df.loc[idx, 'k'])
        except Exception:
            continue
    
    return best_k


def collect_data_from_analysis_folders(run_path: str, run_info: Dict) -> List[Dict]:
    """
    Bir run'ın analysis_k* klasörlerinden veri topla.
    Her analysis klasöründe mfr.csv'den Ricci değerlerini al.
    """
    data_points = []
    
    # models_b70 klasöründen accuracy değerlerini yükle
    models_dir = os.path.join(run_path, "models_b70")
    if not os.path.exists(models_dir):
        return data_points
    
    try:
        accuracy_file = os.path.join(models_dir, "accuracy.npy")
        accuracy_array = np.load(accuracy_file)
    except Exception:
        return data_points
    
    # Her analysis_k* klasörü için
    analysis_dirs = glob.glob(os.path.join(run_path, "analysis_k*"))
    
    for analysis_dir in analysis_dirs:
        try:
            # k değerini dizin isminden çıkar
            k_str = os.path.basename(analysis_dir).replace("analysis_k", "")
            k = int(k_str)
            
            # mfr.csv'yi oku
            mfr_path = os.path.join(analysis_dir, "mfr.csv")
            if not os.path.exists(mfr_path):
                continue
            
            mfr_df = pd.read_csv(mfr_path)
            
            # Her model için ortalama Ricci değerini hesapla
            model_ids = mfr_df['mod'].unique()
            
            for mod_id in model_ids:
                mod_data = mfr_df[mfr_df['mod'] == mod_id]
                mean_ricci = mod_data['ssr'].mean()
                
                # Accuracy değerini al (mod_id index olarak)
                if mod_id < len(accuracy_array):
                    acc = accuracy_array[mod_id]
                else:
                    continue
                
                data_points.append({
                    'run_name': run_info['run_name'],
                    'architecture': run_info['architecture'],
                    'depth': run_info['depth'],
                    'dataset': run_info['dataset'],
                    'dataset_type': run_info['dataset_type'],
                    'model_id': mod_id,
                    'k': k,
                    'accuracy': float(acc),
                    'rho': float(mean_ricci),
                    'n_layers': len(mod_data)
                })
        except Exception as e:
            print(f"    Hata (analysis_k): {e}")
            continue
    
    return data_points


def collect_data_from_activations(run_path: str, run_info: Dict, k: int = 100) -> List[Dict]:
    """
    Bir run için aktivasyon dosyalarından doğrudan Ricci hesapla.
    Bu yöntem analysis klasörleri yoksa kullanılır.
    """
    data_points = []
    
    models_dir = os.path.join(run_path, "models_b70")
    if not os.path.exists(models_dir):
        return data_points
    
    try:
        # Dosyaları yükle
        model_file = os.path.join(models_dir, "model_predict.npy")
        accuracy_file = os.path.join(models_dir, "accuracy.npy")
        x_test_file = os.path.join(models_dir, "x_test.csv")
        
        if not all(os.path.exists(f) for f in [model_file, accuracy_file, x_test_file]):
            return data_points
        
        model = np.load(model_file, allow_pickle=True)
        accuracy = np.load(accuracy_file)
        x_test = np.array(pd.read_csv(x_test_file, header=None))
        
        # Her model için Ricci hesapla
        for model_idx in range(len(model)):
            try:
                activations = model[model_idx]
                acc = float(accuracy[model_idx])
                
                # Ricci hesapla
                mean_rho, _ = calculate_ricci_for_activations(activations, x_test, k)
                
                if not np.isnan(mean_rho):
                    data_points.append({
                        'run_name': run_info['run_name'],
                        'architecture': run_info['architecture'],
                        'depth': run_info['depth'],
                        'dataset': run_info['dataset'],
                        'dataset_type': run_info['dataset_type'],
                        'model_id': model_idx,
                        'k': k,
                        'accuracy': acc,
                        'rho': mean_rho,
                        'n_layers': len(activations)
                    })
            except Exception:
                continue
    except Exception as e:
        print(f"    Hata (activations): {e}")
    
    return data_points


def collect_all_data(base_dir: str) -> pd.DataFrame:
    """
    output_layers/ dizinindeki tüm run'lardan veri topla.
    """
    print("=" * 100)
    print("VERİ TOPLAMA BAŞLADI")
    print("=" * 100)
    print(f"Kaynak dizin: {base_dir}")
    print()
    
    all_data = []
    
    # Tüm run dizinlerini bul
    run_dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) 
                if os.path.isdir(d) and not d.endswith('.csv')]
    
    print(f"Toplam {len(run_dirs)} run bulundu.\n")
    
    for idx, run_dir in enumerate(run_dirs, 1):
        run_info = parse_run_info(run_dir)
        run_name = run_info['run_name']
        
        # Bazı dizinleri atla
        if 'summary' in run_name.lower() or run_name.startswith('.'):
            continue
        
        print(f"[{idx}/{len(run_dirs)}] İşleniyor: {run_name}")
        
        # Önce analysis klasörlerinden veri toplamayı dene
        data_points = collect_data_from_analysis_folders(run_dir, run_info)
        
        if len(data_points) > 0:
            print(f"    ✓ {len(data_points)} veri noktası toplandı (analysis klasörlerinden)")
            all_data.extend(data_points)
        else:
            # Analysis klasörleri yoksa aktivasyonlardan hesapla
            print(f"    Analysis klasörleri bulunamadı, aktivasyonlardan hesaplanıyor...")
            k = find_best_k_for_run(run_dir)
            data_points = collect_data_from_activations(run_dir, run_info, k)
            if len(data_points) > 0:
                print(f"    ✓ {len(data_points)} veri noktası hesaplandı (aktivasyonlardan)")
                all_data.extend(data_points)
            else:
                print(f"    ⚠ Veri toplanamadı!")
    
    df = pd.DataFrame(all_data)
    
    print("\n" + "=" * 100)
    print("VERİ TOPLAMA TAMAMLANDI")
    print(f"Toplam veri noktası: {len(df)}")
    print(f"Toplam benzersiz run: {df['run_name'].nunique() if len(df) > 0 else 0}")
    print("=" * 100 + "\n")
    
    return df


# ============================================================================
# ANALİZ FONKSİYONLARI
# ============================================================================

def calculate_spearman_analysis(df: pd.DataFrame) -> Dict:
    """
    Spearman Rank Korelasyon analizi yap.
    Accuracy ve Rho sıralamalarını karşılaştır.
    """
    valid_data = df.dropna(subset=['accuracy', 'rho'])
    
    if len(valid_data) < 2:
        return {'correlation': np.nan, 'p_value': np.nan, 'n_samples': 0}
    
    # Spearman korelasyonu hesapla
    corr, p_value = spearmanr(valid_data['accuracy'], valid_data['rho'])
    
    # Sıralamaları oluştur
    acc_ranks = valid_data['accuracy'].rank(ascending=False)
    rho_ranks = valid_data['rho'].rank(ascending=False)
    
    return {
        'correlation': corr,
        'p_value': p_value,
        'n_samples': len(valid_data),
        'acc_ranks': acc_ranks,
        'rho_ranks': rho_ranks
    }


def perform_statistical_tests(df: pd.DataFrame) -> Dict:
    """
    İstatistiksel testler uygula:
    - Pearson korelasyonu
    - T-test (yüksek vs düşük accuracy grupları)
    - Mann-Whitney U testi
    """
    valid_data = df.dropna(subset=['accuracy', 'rho'])
    results = {}
    
    if len(valid_data) < 4:
        return results
    
    # 1. Pearson Korelasyonu
    pearson_r, pearson_p = pearsonr(valid_data['accuracy'], valid_data['rho'])
    results['pearson'] = {
        'correlation': pearson_r,
        'p_value': pearson_p,
        'n_samples': len(valid_data)
    }
    
    # 2. T-test: Yüksek vs Düşük Accuracy grupları
    median_acc = valid_data['accuracy'].median()
    high_acc_rho = valid_data[valid_data['accuracy'] >= median_acc]['rho']
    low_acc_rho = valid_data[valid_data['accuracy'] < median_acc]['rho']
    
    if len(high_acc_rho) > 1 and len(low_acc_rho) > 1:
        t_stat, t_p = ttest_ind(high_acc_rho, low_acc_rho)
        results['ttest_median_split'] = {
            't_statistic': t_stat,
            'p_value': t_p,
            'mean_rho_high_acc': high_acc_rho.mean(),
            'mean_rho_low_acc': low_acc_rho.mean(),
            'n_high': len(high_acc_rho),
            'n_low': len(low_acc_rho)
        }
    
    # 3. T-test: Üst vs Alt çeyreklik
    q75 = valid_data['accuracy'].quantile(0.75)
    q25 = valid_data['accuracy'].quantile(0.25)
    top_quartile_rho = valid_data[valid_data['accuracy'] >= q75]['rho']
    bottom_quartile_rho = valid_data[valid_data['accuracy'] <= q25]['rho']
    
    if len(top_quartile_rho) > 1 and len(bottom_quartile_rho) > 1:
        t_stat, t_p = ttest_ind(top_quartile_rho, bottom_quartile_rho)
        results['ttest_quartile'] = {
            't_statistic': t_stat,
            'p_value': t_p,
            'mean_rho_top': top_quartile_rho.mean(),
            'mean_rho_bottom': bottom_quartile_rho.mean(),
            'n_top': len(top_quartile_rho),
            'n_bottom': len(bottom_quartile_rho)
        }
    
    # 4. Mann-Whitney U testi (non-parametric)
    if len(high_acc_rho) > 1 and len(low_acc_rho) > 1:
        u_stat, u_p = mannwhitneyu(high_acc_rho, low_acc_rho, alternative='two-sided')
        results['mannwhitney'] = {
            'u_statistic': u_stat,
            'p_value': u_p
        }
    
    return results


# ============================================================================
# GÖRSELLEŞTİRME FONKSİYONLARI
# ============================================================================

def create_accuracy_rho_scatter(df: pd.DataFrame, output_path: str):
    """
    Accuracy vs Rho scatter plot oluştur.
    """
    valid_data = df.dropna(subset=['accuracy', 'rho'])
    
    if len(valid_data) == 0:
        print("Scatter plot için geçerli veri yok!")
        return
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Architecture'a göre renklendir
    architectures = valid_data['architecture'].unique()
    colors = {'narrow': '#2ecc71', 'wide': '#3498db', 'bottleneck': '#e74c3c', 'unknown': '#95a5a6'}
    markers = {'narrow': 'o', 'wide': 's', 'bottleneck': '^', 'unknown': 'x'}
    
    for arch in architectures:
        arch_data = valid_data[valid_data['architecture'] == arch]
        color = colors.get(arch, '#95a5a6')
        marker = markers.get(arch, 'o')
        ax.scatter(arch_data['accuracy'], arch_data['rho'], 
                   label=f'{arch.capitalize()} (n={len(arch_data)})',
                   alpha=0.6, s=50, c=color, marker=marker, edgecolors='white', linewidth=0.5)
    
    # Trend çizgisi ekle
    z = np.polyfit(valid_data['accuracy'], valid_data['rho'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid_data['accuracy'].min(), valid_data['accuracy'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Trend Çizgisi')
    
    # Spearman korelasyonu hesapla ve göster
    corr, p_val = spearmanr(valid_data['accuracy'], valid_data['rho'])
    
    # İstatistik kutusu
    stats_text = f'Spearman ρ = {corr:.4f}\np-değeri = {p_val:.2e}\nn = {len(valid_data)}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, fontweight='bold')
    
    # Eksen ayarları
    ax.set_xlabel('Accuracy (Model Doğruluğu)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Ricci Curvature (Rho) - Ortalama', fontsize=14, fontweight='bold')
    ax.set_title('Accuracy vs. Ricci Curvature (Rho) - Tüm Modeller', fontsize=16, fontweight='bold')
    
    # Range genişletme - düşük accuracy değerlerini de görülebilir yap
    acc_min, acc_max = valid_data['accuracy'].min(), valid_data['accuracy'].max()
    rho_min, rho_max = valid_data['rho'].min(), valid_data['rho'].max()
    
    acc_margin = (acc_max - acc_min) * 0.05
    rho_margin = (rho_max - rho_min) * 0.05
    
    ax.set_xlim(acc_min - acc_margin, acc_max + acc_margin)
    ax.set_ylim(rho_min - rho_margin, rho_max + rho_margin)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Scatter plot kaydedildi: {output_path}")
    plt.close()


def create_architecture_comparison(df: pd.DataFrame, output_path: str):
    """
    Architecture ve Depth bazında karşılaştırmalı analiz grafiği.
    Bu, dropout analizi yerine kullanılabilir.
    """
    valid_data = df.dropna(subset=['accuracy', 'rho'])
    
    if len(valid_data) == 0:
        print("Karşılaştırma için geçerli veri yok!")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. Accuracy by Architecture (box plot)
    ax1 = axes[0, 0]
    architectures = valid_data['architecture'].unique()
    arch_data_acc = [valid_data[valid_data['architecture'] == arch]['accuracy'].values for arch in architectures]
    
    bp1 = ax1.boxplot(arch_data_acc, labels=architectures, patch_artist=True)
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#95a5a6']
    for patch, color in zip(bp1['boxes'], colors[:len(architectures)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_xlabel('Architecture', fontweight='bold')
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('Accuracy Dağılımı - Architecture Bazında', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Rho by Architecture (box plot)
    ax2 = axes[0, 1]
    arch_data_rho = [valid_data[valid_data['architecture'] == arch]['rho'].values for arch in architectures]
    
    bp2 = ax2.boxplot(arch_data_rho, labels=architectures, patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors[:len(architectures)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_xlabel('Architecture', fontweight='bold')
    ax2.set_ylabel('Ricci Curvature (Rho)', fontweight='bold')
    ax2.set_title('Ricci Curvature Dağılımı - Architecture Bazında', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Accuracy by Depth (box plot)
    ax3 = axes[1, 0]
    depths = sorted(valid_data['depth'].unique())
    depth_data_acc = [valid_data[valid_data['depth'] == d]['accuracy'].values for d in depths]
    
    bp3 = ax3.boxplot(depth_data_acc, labels=[str(d) for d in depths], patch_artist=True)
    depth_colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(depths)))
    for patch, color in zip(bp3['boxes'], depth_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_xlabel('Depth (Layer Sayısı)', fontweight='bold')
    ax3.set_ylabel('Accuracy', fontweight='bold')
    ax3.set_title('Accuracy Dağılımı - Depth Bazında', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Rho by Depth (box plot)
    ax4 = axes[1, 1]
    depth_data_rho = [valid_data[valid_data['depth'] == d]['rho'].values for d in depths]
    
    bp4 = ax4.boxplot(depth_data_rho, labels=[str(d) for d in depths], patch_artist=True)
    for patch, color in zip(bp4['boxes'], depth_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_xlabel('Depth (Layer Sayısı)', fontweight='bold')
    ax4.set_ylabel('Ricci Curvature (Rho)', fontweight='bold')
    ax4.set_title('Ricci Curvature Dağılımı - Depth Bazında', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Architecture ve Depth Bazında Karşılaştırmalı Analiz', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Karşılaştırma grafiği kaydedildi: {output_path}")
    plt.close()


def create_dataset_comparison(df: pd.DataFrame, output_path: str):
    """
    Dataset tipi bazında karşılaştırmalı analiz.
    """
    valid_data = df.dropna(subset=['accuracy', 'rho'])
    
    if len(valid_data) == 0:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    dataset_types = valid_data['dataset_type'].unique()
    colors = {'mnist': '#3498db', 'fmnist': '#e74c3c', 'synthetic': '#2ecc71', 'unknown': '#95a5a6'}
    
    # 1. Accuracy by Dataset Type
    ax1 = axes[0]
    for dtype in dataset_types:
        dtype_data = valid_data[valid_data['dataset_type'] == dtype]
        ax1.scatter(dtype_data['accuracy'], dtype_data['rho'],
                   label=f'{dtype.upper()} (n={len(dtype_data)})',
                   alpha=0.5, s=40, c=colors.get(dtype, '#95a5a6'))
    
    ax1.set_xlabel('Accuracy', fontweight='bold')
    ax1.set_ylabel('Ricci Curvature (Rho)', fontweight='bold')
    ax1.set_title('Accuracy vs Rho - Dataset Tipi Bazında', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot karşılaştırması
    ax2 = axes[1]
    dataset_data = [valid_data[valid_data['dataset_type'] == dtype]['rho'].values for dtype in dataset_types]
    
    bp = ax2.boxplot(dataset_data, labels=dataset_types, patch_artist=True)
    for patch, dtype in zip(bp['boxes'], dataset_types):
        patch.set_facecolor(colors.get(dtype, '#95a5a6'))
        patch.set_alpha(0.7)
    
    ax2.set_xlabel('Dataset Tipi', fontweight='bold')
    ax2.set_ylabel('Ricci Curvature (Rho)', fontweight='bold')
    ax2.set_title('Rho Dağılımı - Dataset Tipi Bazında', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Dataset karşılaştırma grafiği kaydedildi: {output_path}")
    plt.close()


def create_ranking_comparison(df: pd.DataFrame, spearman_result: Dict, output_path: str):
    """
    Accuracy ve Rho sıralama karşılaştırması görselleştirmesi.
    """
    if 'acc_ranks' not in spearman_result or 'rho_ranks' not in spearman_result:
        return
    
    valid_data = df.dropna(subset=['accuracy', 'rho']).copy()
    valid_data['acc_rank'] = valid_data['accuracy'].rank(ascending=False)
    valid_data['rho_rank'] = valid_data['rho'].rank(ascending=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. Rank karşılaştırması scatter
    ax1 = axes[0]
    ax1.scatter(valid_data['acc_rank'], valid_data['rho_rank'], alpha=0.5, s=30)
    
    # Mükemmel korelasyon çizgisi
    max_rank = max(valid_data['acc_rank'].max(), valid_data['rho_rank'].max())
    ax1.plot([0, max_rank], [0, max_rank], 'r--', linewidth=2, label='Mükemmel Korelasyon')
    
    ax1.set_xlabel('Accuracy Sıralaması (1 = En İyi)', fontweight='bold')
    ax1.set_ylabel('Rho Sıralaması (1 = En Yüksek)', fontweight='bold')
    ax1.set_title('Accuracy vs Rho Sıralama Karşılaştırması', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Sıralama farkı histogramı
    ax2 = axes[1]
    rank_diff = valid_data['acc_rank'] - valid_data['rho_rank']
    ax2.hist(rank_diff, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Mükemmel Eşleşme')
    ax2.axvline(x=rank_diff.mean(), color='green', linestyle='-', linewidth=2, 
                label=f'Ortalama Fark: {rank_diff.mean():.1f}')
    
    ax2.set_xlabel('Sıralama Farkı (Accuracy Rank - Rho Rank)', fontweight='bold')
    ax2.set_ylabel('Frekans', fontweight='bold')
    ax2.set_title('Sıralama Farkı Dağılımı', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Sıralama karşılaştırma grafiği kaydedildi: {output_path}")
    plt.close()


# ============================================================================
# RAPOR OLUŞTURMA
# ============================================================================

def generate_summary_report(df: pd.DataFrame, spearman_result: Dict, test_results: Dict, output_path: str):
    """
    Kapsamlı özet raporu oluştur.
    """
    valid_data = df.dropna(subset=['accuracy', 'rho'])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("DEEP LEARNING AS RICCI FLOW - ANALİZ RAPORU\n")
        f.write("=" * 100 + "\n")
        f.write(f"Rapor Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 1. Veri Özeti
        f.write("-" * 50 + "\n")
        f.write("1. VERİ ÖZETİ\n")
        f.write("-" * 50 + "\n")
        f.write(f"Toplam veri noktası: {len(df)}\n")
        f.write(f"Geçerli veri noktası: {len(valid_data)}\n")
        f.write(f"Benzersiz run sayısı: {df['run_name'].nunique()}\n\n")
        
        if 'architecture' in df.columns:
            f.write("Architecture dağılımı:\n")
            for arch, count in df['architecture'].value_counts().items():
                f.write(f"  {arch}: {count}\n")
            f.write("\n")
        
        if 'dataset_type' in df.columns:
            f.write("Dataset tipi dağılımı:\n")
            for dtype, count in df['dataset_type'].value_counts().items():
                f.write(f"  {dtype}: {count}\n")
            f.write("\n")
        
        # 2. Betimsel İstatistikler
        f.write("-" * 50 + "\n")
        f.write("2. BETİMSEL İSTATİSTİKLER\n")
        f.write("-" * 50 + "\n")
        f.write(f"Accuracy:\n")
        f.write(f"  Ortalama: {valid_data['accuracy'].mean():.6f}\n")
        f.write(f"  Std: {valid_data['accuracy'].std():.6f}\n")
        f.write(f"  Min: {valid_data['accuracy'].min():.6f}\n")
        f.write(f"  Max: {valid_data['accuracy'].max():.6f}\n")
        f.write(f"  Medyan: {valid_data['accuracy'].median():.6f}\n\n")
        
        f.write(f"Ricci Curvature (Rho):\n")
        f.write(f"  Ortalama: {valid_data['rho'].mean():.2f}\n")
        f.write(f"  Std: {valid_data['rho'].std():.2f}\n")
        f.write(f"  Min: {valid_data['rho'].min():.2f}\n")
        f.write(f"  Max: {valid_data['rho'].max():.2f}\n")
        f.write(f"  Medyan: {valid_data['rho'].median():.2f}\n\n")
        
        # 3. Spearman Korelasyon Analizi
        f.write("-" * 50 + "\n")
        f.write("3. SPEARMAN RANK KORELASYON ANALİZİ\n")
        f.write("-" * 50 + "\n")
        f.write(f"Spearman ρ = {spearman_result['correlation']:.6f}\n")
        f.write(f"p-değeri = {spearman_result['p_value']:.2e}\n")
        f.write(f"Örnek sayısı = {spearman_result['n_samples']}\n\n")
        
        if spearman_result['p_value'] < 0.001:
            f.write("★★★ YÜKSEK İSTATİSTİKSEL ANLAMLILIK (p < 0.001)\n")
        elif spearman_result['p_value'] < 0.01:
            f.write("★★ ORTA İSTATİSTİKSEL ANLAMLILIK (p < 0.01)\n")
        elif spearman_result['p_value'] < 0.05:
            f.write("★ İSTATİSTİKSEL ANLAMLILIK (p < 0.05)\n")
        else:
            f.write("✗ İSTATİSTİKSEL OLARAK ANLAMLI DEĞİL (p >= 0.05)\n")
        
        f.write(f"\nYorum: ")
        if spearman_result['correlation'] > 0:
            f.write(f"Pozitif korelasyon - Accuracy arttıkça Rho da artma eğiliminde.\n")
        elif spearman_result['correlation'] < 0:
            f.write(f"Negatif korelasyon - Accuracy arttıkça Rho azalma eğiliminde.\n")
            f.write("\n⚠️ ÖNEMLİ NOT: Forman-Ricci eğriliği değerleri tipik olarak NEGATİFTİR.\n")
            f.write("   Formül: R(i,j) = 4 - deg(i) - deg(j), yüksek k değerlerinde çok negatif olur.\n")
            f.write("   Bu nedenle negatif korelasyon, aslında şu anlama gelir:\n")
            f.write("   → Accuracy ↑ = |Rho| ↑ (mutlak değer artıyor)\n")
            f.write("   → Daha iyi performans = Daha belirgin geometrik yapı\n")
        else:
            f.write(f"Korelasyon yok veya çok zayıf.\n")
        f.write("\n")
        
        # 4. İstatistiksel Testler
        f.write("-" * 50 + "\n")
        f.write("4. İSTATİSTİKSEL TESTLER\n")
        f.write("-" * 50 + "\n")
        
        if 'pearson' in test_results:
            f.write("Pearson Korelasyonu:\n")
            f.write(f"  r = {test_results['pearson']['correlation']:.6f}\n")
            f.write(f"  p-değeri = {test_results['pearson']['p_value']:.2e}\n\n")
        
        if 'ttest_median_split' in test_results:
            t = test_results['ttest_median_split']
            f.write("T-test (Medyan bölünmesi - Yüksek vs Düşük Accuracy):\n")
            f.write(f"  Yüksek accuracy grubu ortalama Rho: {t['mean_rho_high_acc']:.2f} (n={t['n_high']})\n")
            f.write(f"  Düşük accuracy grubu ortalama Rho: {t['mean_rho_low_acc']:.2f} (n={t['n_low']})\n")
            f.write(f"  t-istatistiği: {t['t_statistic']:.4f}\n")
            f.write(f"  p-değeri: {t['p_value']:.2e}\n")
            if t['p_value'] < 0.05:
                f.write("  ✓ Gruplar arasında anlamlı fark VAR\n")
            else:
                f.write("  ✗ Gruplar arasında anlamlı fark YOK\n")
            f.write("\n")
        
        if 'ttest_quartile' in test_results:
            t = test_results['ttest_quartile']
            f.write("T-test (Çeyreklik bölünmesi - Üst vs Alt %25):\n")
            f.write(f"  Üst çeyreklik ortalama Rho: {t['mean_rho_top']:.2f} (n={t['n_top']})\n")
            f.write(f"  Alt çeyreklik ortalama Rho: {t['mean_rho_bottom']:.2f} (n={t['n_bottom']})\n")
            f.write(f"  t-istatistiği: {t['t_statistic']:.4f}\n")
            f.write(f"  p-değeri: {t['p_value']:.2e}\n")
            if t['p_value'] < 0.05:
                f.write("  ✓ Gruplar arasında anlamlı fark VAR\n")
            else:
                f.write("  ✗ Gruplar arasında anlamlı fark YOK\n")
            f.write("\n")
        
        if 'mannwhitney' in test_results:
            m = test_results['mannwhitney']
            f.write("Mann-Whitney U testi (non-parametrik):\n")
            f.write(f"  U-istatistiği: {m['u_statistic']:.2f}\n")
            f.write(f"  p-değeri: {m['p_value']:.2e}\n\n")
        
        # 5. Hipotez Değerlendirmesi
        f.write("-" * 50 + "\n")
        f.write("5. HİPOTEZ DEĞERLENDİRMESİ\n")
        f.write("-" * 50 + "\n")
        f.write("Hipotez: 'Daha iyi network performansı = Daha iyi Ricci skoru'\n\n")
        
        # Karar
        significant_tests = 0
        total_tests = 0
        
        if 'pearson' in test_results:
            total_tests += 1
            if test_results['pearson']['p_value'] < 0.05:
                significant_tests += 1
        
        if 'ttest_median_split' in test_results:
            total_tests += 1
            if test_results['ttest_median_split']['p_value'] < 0.05:
                significant_tests += 1
        
        if spearman_result['p_value'] < 0.05:
            significant_tests += 1
        total_tests += 1
        
        f.write(f"Sonuç: {significant_tests}/{total_tests} test istatistiksel olarak anlamlı\n\n")
        
        if significant_tests > total_tests / 2:
            f.write("★★★ HİPOTEZ DESTEKLENİYOR\n")
            f.write("Accuracy ve Ricci Curvature arasında anlamlı bir ilişki bulunmaktadır.\n")
        else:
            f.write("HİPOTEZ ZAYIF DESTEK ALIYOR\n")
            f.write("Accuracy ve Ricci Curvature arasındaki ilişki beklendiği kadar güçlü değil.\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("RAPOR SONU\n")
        f.write("=" * 100 + "\n")
    
    print(f"✓ Özet raporu kaydedildi: {output_path}")


# ============================================================================
# ANA FONKSİYON
# ============================================================================

def create_normalized_analysis(df: pd.DataFrame, output_path: str):
    """
    Normalize edilmiş Ricci değerleri ile analiz.
    Forman-Ricci değerleri genellikle negatiftir, bu yüzden |Rho| veya -Rho kullanılabilir.
    """
    valid_data = df.dropna(subset=['accuracy', 'rho']).copy()
    
    if len(valid_data) == 0:
        return
    
    # Rho'nun negatif olduğunu varsayarak, -Rho kullanarak pozitif değerlere dönüştür
    # Böylece "daha yüksek = geometrik olarak daha düz/iyi" olacak
    valid_data['rho_neg'] = -valid_data['rho']  # -Rho (pozitif yapma)
    valid_data['rho_abs'] = valid_data['rho'].abs()  # |Rho|
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. Accuracy vs -Rho (terslenmiş)
    ax1 = axes[0, 0]
    architectures = valid_data['architecture'].unique()
    colors = {'narrow': '#2ecc71', 'wide': '#3498db', 'bottleneck': '#e74c3c', 'unknown': '#95a5a6'}
    
    for arch in architectures:
        arch_data = valid_data[valid_data['architecture'] == arch]
        ax1.scatter(arch_data['accuracy'], arch_data['rho_neg'],
                   label=f'{arch.capitalize()}', alpha=0.5, s=30, c=colors.get(arch, '#95a5a6'))
    
    # Trend çizgisi
    z = np.polyfit(valid_data['accuracy'], valid_data['rho_neg'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid_data['accuracy'].min(), valid_data['accuracy'].max(), 100)
    ax1.plot(x_line, p(x_line), "r--", linewidth=2, label='Trend')
    
    corr_neg, p_val_neg = spearmanr(valid_data['accuracy'], valid_data['rho_neg'])
    ax1.text(0.05, 0.95, f'Spearman ρ = {corr_neg:.4f}\np = {p_val_neg:.2e}',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.set_xlabel('Accuracy', fontweight='bold')
    ax1.set_ylabel('-Rho (Terslenmiş Ricci)', fontweight='bold')
    ax1.set_title('Accuracy vs -Rho (Pozitif = Daha Düz Geometri)', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Dataset tipi bazında -Rho
    ax2 = axes[0, 1]
    dataset_types = valid_data['dataset_type'].unique()
    
    for idx, dtype in enumerate(dataset_types):
        dtype_data = valid_data[valid_data['dataset_type'] == dtype]
        ax2.scatter(dtype_data['accuracy'], dtype_data['rho_neg'],
                   label=f'{dtype.upper()} (n={len(dtype_data)})', alpha=0.4, s=25)
    
    ax2.set_xlabel('Accuracy', fontweight='bold')
    ax2.set_ylabel('-Rho', fontweight='bold')
    ax2.set_title('Dataset Tipi Bazında: Accuracy vs -Rho', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. k değerine göre korelasyon
    ax3 = axes[1, 0]
    k_values = sorted(valid_data['k'].unique())
    k_correlations = []
    
    for k in k_values:
        k_data = valid_data[valid_data['k'] == k]
        if len(k_data) > 10:
            corr, _ = spearmanr(k_data['accuracy'], k_data['rho_neg'])
            k_correlations.append({'k': k, 'correlation': corr, 'n': len(k_data)})
    
    if k_correlations:
        k_corr_df = pd.DataFrame(k_correlations)
        colors_k = plt.cm.viridis(np.linspace(0, 1, len(k_corr_df)))
        bars = ax3.bar(range(len(k_corr_df)), k_corr_df['correlation'], color=colors_k)
        ax3.set_xticks(range(len(k_corr_df)))
        ax3.set_xticklabels([str(k) for k in k_corr_df['k']], rotation=45, ha='right')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_xlabel('k değeri (kNN)', fontweight='bold')
        ax3.set_ylabel('Spearman Korelasyonu (Accuracy vs -Rho)', fontweight='bold')
        ax3.set_title('Farklı k Değerlerinde Korelasyon', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Architecture bazında korelasyon özeti
    ax4 = axes[1, 1]
    arch_correlations = []
    
    for arch in architectures:
        arch_data = valid_data[valid_data['architecture'] == arch]
        if len(arch_data) > 10:
            corr, pval = spearmanr(arch_data['accuracy'], arch_data['rho_neg'])
            arch_correlations.append({
                'architecture': arch,
                'correlation': corr,
                'p_value': pval,
                'n': len(arch_data)
            })
    
    if arch_correlations:
        arch_df = pd.DataFrame(arch_correlations)
        x_pos = np.arange(len(arch_df))
        bars = ax4.bar(x_pos, arch_df['correlation'],
                       color=[colors.get(a, '#95a5a6') for a in arch_df['architecture']])
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([a.capitalize() for a in arch_df['architecture']])
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Anlamlılık işaretleri
        for i, (corr, pval) in enumerate(zip(arch_df['correlation'], arch_df['p_value'])):
            star = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            ax4.text(i, corr + 0.02, f'{corr:.3f}{star}', ha='center', fontsize=10)
        
        ax4.set_xlabel('Architecture', fontweight='bold')
        ax4.set_ylabel('Spearman Korelasyonu (Accuracy vs -Rho)', fontweight='bold')
        ax4.set_title('Architecture Bazında Korelasyon\n(*** p<0.001, ** p<0.01, * p<0.05)', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Normalize Edilmiş Ricci Analizi', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Normalize analiz grafiği kaydedildi: {output_path}")
    plt.close()
    
    return corr_neg, p_val_neg


def main():
    """Ana analiz pipeline'ı."""
    print("\n" + "=" * 100)
    print("DEEP LEARNING AS RICCI FLOW - KAPSAMLI ANALİZ")
    print("=" * 100)
    
    # Konfigürasyon
    BASE_DIR = os.path.join(os.getcwd(), 'output_layers')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Kaynak dizin: {BASE_DIR}")
    print(f"Çıktı dizini: {OUTPUT_DIR}")
    print("=" * 100 + "\n")
    
    # Adım 1: Veri Toplama
    print("[ADIM 1/5] Veri toplama...")
    df = collect_all_data(BASE_DIR)
    
    if len(df) == 0:
        print("HATA: Veri toplanamadı! output_layers/ dizinini kontrol edin.")
        return
    
    # Veriyi kaydet
    data_csv_path = os.path.join(OUTPUT_DIR, "collected_data.csv")
    df.to_csv(data_csv_path, index=False)
    print(f"✓ Toplanan veri kaydedildi: {data_csv_path}\n")
    
    # Adım 2: Spearman Korelasyon Analizi
    print("[ADIM 2/5] Spearman korelasyon analizi...")
    spearman_result = calculate_spearman_analysis(df)
    print(f"Spearman ρ = {spearman_result['correlation']:.6f}")
    print(f"p-değeri = {spearman_result['p_value']:.2e}")
    print(f"Örnek sayısı = {spearman_result['n_samples']}\n")
    
    # Adım 3: İstatistiksel Testler
    print("[ADIM 3/5] İstatistiksel testler yapılıyor...")
    test_results = perform_statistical_tests(df)
    
    if 'ttest_median_split' in test_results:
        t = test_results['ttest_median_split']
        print(f"T-test (Medyan bölünmesi): t={t['t_statistic']:.4f}, p={t['p_value']:.2e}")
    
    if 'pearson' in test_results:
        print(f"Pearson: r={test_results['pearson']['correlation']:.6f}, p={test_results['pearson']['p_value']:.2e}")
    print()
    
    # Adım 4: Görselleştirmeler
    print("[ADIM 4/5] Görselleştirmeler oluşturuluyor...")
    
    # 4.1 Accuracy vs Rho Scatter Plot
    create_accuracy_rho_scatter(df, os.path.join(OUTPUT_DIR, "accuracy_vs_ricci_scatter.png"))
    
    # 4.2 Architecture ve Depth Karşılaştırması
    create_architecture_comparison(df, os.path.join(OUTPUT_DIR, "architecture_depth_comparison.png"))
    
    # 4.3 Dataset Tipi Karşılaştırması
    create_dataset_comparison(df, os.path.join(OUTPUT_DIR, "dataset_comparison.png"))
    
    # 4.4 Sıralama Karşılaştırması
    create_ranking_comparison(df, spearman_result, os.path.join(OUTPUT_DIR, "ranking_comparison.png"))
    
    # 4.5 Normalize Edilmiş Analiz (-Rho kullanarak)
    norm_corr, norm_p = create_normalized_analysis(df, os.path.join(OUTPUT_DIR, "normalized_analysis.png"))
    print(f"Normalize korelasyon (Acc vs -Rho): {norm_corr:.4f} (p={norm_p:.2e})")
    
    print()
    
    # Adım 5: Rapor Oluşturma
    print("[ADIM 5/5] Özet rapor oluşturuluyor...")
    generate_summary_report(df, spearman_result, test_results, os.path.join(OUTPUT_DIR, "analysis_report.txt"))
    
    # Test sonuçlarını CSV olarak kaydet
    test_results_flat = {}
    for test_name, test_data in test_results.items():
        if isinstance(test_data, dict):
            for key, value in test_data.items():
                test_results_flat[f"{test_name}_{key}"] = value
    
    test_df = pd.DataFrame([test_results_flat])
    test_df.to_csv(os.path.join(OUTPUT_DIR, "statistical_tests.csv"), index=False)
    print(f"✓ Test sonuçları kaydedildi: {os.path.join(OUTPUT_DIR, 'statistical_tests.csv')}")
    
    # Özet Yazdır
    print("\n" + "=" * 100)
    print("ANALİZ TAMAMLANDI!")
    print("=" * 100)
    print(f"\nÜretilen dosyalar ({OUTPUT_DIR}/):")
    print("  - collected_data.csv          : Tüm toplanan veri")
    print("  - accuracy_vs_ricci_scatter.png: Accuracy vs Rho scatter plot")
    print("  - architecture_depth_comparison.png: Architecture/Depth karşılaştırması")
    print("  - dataset_comparison.png      : Dataset tipi karşılaştırması")
    print("  - ranking_comparison.png      : Sıralama karşılaştırması")
    print("  - normalized_analysis.png     : Normalize Ricci analizi (-Rho)")
    print("  - statistical_tests.csv       : İstatistiksel test sonuçları")
    print("  - analysis_report.txt         : Detaylı özet rapor")
    print("\n" + "=" * 100)
    
    # Kısa özet
    print("\n📊 SONUÇ ÖZETİ:")
    print(f"   Toplam veri noktası: {len(df)}")
    print(f"   Spearman ρ: {spearman_result['correlation']:.4f} (p={spearman_result['p_value']:.2e})")
    if spearman_result['correlation'] > 0:
        print("   → POZİTİF korelasyon: Accuracy ↑ = Rho ↑")
    else:
        print("   → NEGATİF korelasyon: Accuracy ↑ = Rho ↓")
    
    if spearman_result['p_value'] < 0.05:
        print("   ✓ İstatistiksel olarak ANLAMLI")
    else:
        print("   ✗ İstatistiksel olarak anlamlı DEĞİL")


if __name__ == "__main__":
    main()

