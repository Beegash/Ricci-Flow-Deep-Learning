#!/usr/bin/env python3
"""
Deep Learning as Ricci Flow - Comprehensive Analysis Script

Bu script, output_layers/ dizinindeki tüm aktivasyon dosyalarını işleyerek:
1. Her run için Accuracy ve Ricci Curvature (Rho) değerlerini hesaplar
2. Spearman Rank Correlation analizi yapar
3. Accuracy vs Rho görselleştirmeleri oluşturur
4. Dropout rate analizi yapar
5. Parametrik testler uygular
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, ttest_ind
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import triu as sp_triu
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# RICCI CURVATURE CALCULATION FUNCTIONS
# ============================================================================

def build_knn_graph(X: np.ndarray, k: int) -> csr_matrix:
    """Return an undirected, unweighted kNN adjacency in CSR format."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.dtype != np.float32 and X.dtype != np.float64:
        X = X.astype(np.float32, copy=False)
    
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(X)
    A = knn.kneighbors_graph(X, mode="connectivity")
    A = A.maximum(A.T)  # Make symmetric
    A.setdiag(0)  # Remove self-loops
    A.eliminate_zeros()
    return A.tocsr()


def global_forman_ricci(A: csr_matrix) -> float:
    """
    Calculate Global Forman-Ricci curvature coefficient.
    Formula: R(i,j) = 4 - deg(i) - deg(j) for each edge (i,j)
    Returns: Sum of all edge curvatures (Rho)
    """
    deg = np.asarray(A.sum(axis=1)).ravel()
    A_ut = sp_triu(A, k=1).tocoo()
    curv = 4.0 - deg[A_ut.row] - deg[A_ut.col]
    return float(curv.sum())


def calculate_ricci_for_model(activations: List[np.ndarray], X0: np.ndarray, k: int) -> float:
    """
    Calculate average Ricci curvature across all layers for a single model.
    
    Args:
        activations: List of layer activation arrays
        X0: Original input data (baseline)
        k: Number of neighbors for kNN graph
        
    Returns:
        Average Ricci curvature (Rho) across all layers
    """
    # Calculate Ricci for input layer
    A0 = build_knn_graph(X0, k)
    ric0 = global_forman_ricci(A0)
    
    ric_values = [ric0]
    
    # Calculate Ricci for each activation layer
    for Xl in activations:
        try:
            A = build_knn_graph(np.asarray(Xl), k)
            ric = global_forman_ricci(A)
            ric_values.append(ric)
        except Exception as e:
            print(f"  Warning: Could not calculate Ricci for layer: {e}")
            continue
    
    # Return average Ricci across all layers
    return np.mean(ric_values) if ric_values else np.nan


# ============================================================================
# DATA COLLECTION FUNCTIONS
# ============================================================================

def parse_run_info(run_path: str) -> Dict[str, str]:
    """
    Parse run information from directory path.
    Example: 'bottleneck_5_fmnist_sandals_vs_boots' ->
        {'architecture': 'bottleneck', 'n_layers': '5', 'dataset': 'fmnist_sandals_vs_boots'}
    """
    run_name = os.path.basename(run_path)
    parts = run_name.split('_')
    
    info = {
        'run_name': run_name,
        'full_path': run_path
    }
    
    # Parse architecture type
    if 'bottleneck' in run_name.lower():
        info['architecture'] = 'bottleneck'
    elif 'narrow' in run_name.lower():
        info['architecture'] = 'narrow'
    elif 'wide' in run_name.lower():
        info['architecture'] = 'wide'
    else:
        info['architecture'] = 'unknown'
    
    # Try to extract layer count (usually first number after architecture)
    for i, part in enumerate(parts):
        if part.isdigit():
            info['n_layers'] = part
            break
    
    # Extract dataset name (rest of the name)
    dataset_start = 0
    for i, part in enumerate(parts):
        if part.isdigit():
            dataset_start = i + 1
            break
    info['dataset'] = '_'.join(parts[dataset_start:])
    
    return info


def find_all_runs(base_dir: str) -> List[Dict[str, str]]:
    """Find all model directories in output_layers/."""
    runs = []
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist!")
        return runs
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        
        # Check if it's a directory containing models_b70
        models_dir = os.path.join(item_path, 'models_b70')
        if os.path.isdir(models_dir):
            info = parse_run_info(item_path)
            info['models_dir'] = models_dir
            runs.append(info)
    
    return runs


def load_run_data(models_dir: str, k: int = 100) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load model activations, accuracy, and test data for a run.
    
    Returns:
        (activations, accuracy, x_test) or (None, None, None) if loading fails
    """
    try:
        model_file = os.path.join(models_dir, "model_predict.npy")
        accuracy_file = os.path.join(models_dir, "accuracy.npy")
        x_test_file = os.path.join(models_dir, "x_test.csv")
        
        if not all(os.path.exists(f) for f in [model_file, accuracy_file, x_test_file]):
            return None, None, None
        
        model = np.load(model_file, allow_pickle=True)
        accuracy = np.load(accuracy_file)
        x_test = np.array(pd.read_csv(x_test_file, header=None))
        
        return model, accuracy, x_test
        
    except Exception as e:
        print(f"  Error loading data from {models_dir}: {e}")
        return None, None, None


def load_master_summary(base_dir: str) -> Optional[pd.DataFrame]:
    """Load master grid search summary to get k values and metadata."""
    master_path = os.path.join(base_dir, "MASTER_GRID_SEARCH_SUMMARY.csv")
    if os.path.exists(master_path):
        try:
            df = pd.read_csv(master_path)
            return df
        except Exception as e:
            print(f"Warning: Could not load master summary: {e}")
    return None


def get_k_value_for_run(run_name: str, master_summary: Optional[pd.DataFrame], default_k: int = 100) -> int:
    """
    Get k value for a run from master summary.
    If multiple k values exist, use the one with strongest correlation.
    """
    if master_summary is None:
        return default_k
    
    # Try to match run name to master summary
    # Run name format: "architecture_depth_dataset" (e.g., "narrow_5_mnist_1_vs_7")
    # Master summary format: "architecture", "depth", "dataset" in separate columns
    
    # Extract architecture and dataset from run_name
    parts = run_name.split('_')
    
    # Find matching rows
    matches = master_summary.copy()
    
    # Try to match by architecture
    for arch in ['narrow', 'bottleneck', 'wide']:
        if arch in run_name.lower():
            matches = matches[matches['architecture'].str.contains(arch, case=False, na=False)]
            break
    
    # Try to match by dataset
    for dataset_part in parts:
        if any(ds in dataset_part for ds in ['mnist', 'fmnist', 'synthetic']):
            matches = matches[matches['dataset'].str.contains(dataset_part, case=False, na=False)]
            break
    
    if len(matches) > 0:
        # Use k value with strongest absolute correlation
        matches = matches.copy()
        matches['abs_r_all'] = matches['r_all'].abs()
        best_match = matches.loc[matches['abs_r_all'].idxmax()]
        return int(best_match['k'])
    
    return default_k


def collect_all_runs_data(base_dir: str, k: Optional[int] = None) -> pd.DataFrame:
    """
    Collect Accuracy and Ricci (Rho) values for all runs.
    
    Args:
        base_dir: Base directory containing output_layers/
        k: Optional fixed k value. If None, will try to get from master summary
        
    Returns:
        DataFrame with columns: run_name, architecture, dataset, n_layers,
                                accuracy, rho, k
    """
    print("=" * 100)
    print("COLLECTING DATA FROM ALL RUNS")
    print("=" * 100)
    
    # Load master summary for k values
    master_summary = load_master_summary(base_dir)
    if master_summary is not None:
        print("✓ Loaded master summary for k value lookup")
    else:
        print("⚠ Master summary not found, using default k values")
    
    all_data = []
    runs = find_all_runs(base_dir)
    
    print(f"Found {len(runs)} runs to process...\n")
    
    for idx, run_info in enumerate(runs, 1):
        run_name = run_info['run_name']
        models_dir = run_info['models_dir']
        
        print(f"[{idx}/{len(runs)}] Processing: {run_name}")
        
        # Determine k value for this run
        if k is None:
            run_k = get_k_value_for_run(run_name, master_summary, default_k=100)
            print(f"  Using k={run_k} (from master summary or default)")
        else:
            run_k = k
            print(f"  Using k={run_k} (fixed)")
        
        # Load model data
        model, accuracy, x_test = load_run_data(models_dir, run_k)
        
        if model is None or accuracy is None or x_test is None:
            print(f"  ⚠ Skipping: Missing required files")
            continue
        
        # Calculate Ricci for each model in this run
        print(f"  Processing {len(model)} models...")
        rho_values = []
        acc_values = []
        
        for model_idx in range(len(model)):
            try:
                activations = model[model_idx]
                acc = float(accuracy[model_idx])
                
                # Calculate average Ricci across layers
                rho = calculate_ricci_for_model(activations, x_test, run_k)
                
                if not np.isnan(rho):
                    rho_values.append(rho)
                    acc_values.append(acc)
                    
            except Exception as e:
                print(f"    Warning: Model {model_idx} failed: {e}")
                continue
        
        if len(rho_values) > 0:
            # Store per-model data
            for rho, acc in zip(rho_values, acc_values):
                all_data.append({
                    'run_name': run_name,
                    'architecture': run_info['architecture'],
                    'dataset': run_info['dataset'],
                    'n_layers': run_info.get('n_layers', 'unknown'),
                    'accuracy': acc,
                    'rho': rho,
                    'k': run_k
                })
            
            print(f"  ✓ Processed {len(rho_values)} models successfully")
        else:
            print(f"  ⚠ No valid Ricci values calculated")
        
        print()
    
    df = pd.DataFrame(all_data)
    print(f"\n{'='*100}")
    print(f"DATA COLLECTION COMPLETE")
    print(f"Total runs processed: {len(runs)}")
    print(f"Total data points: {len(df)}")
    print(f"{'='*100}\n")
    
    return df


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def calculate_spearman_correlation(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculate Spearman rank correlation between Accuracy and Rho.
    
    Returns:
        (correlation_coefficient, p_value)
    """
    valid_data = df.dropna(subset=['accuracy', 'rho'])
    
    if len(valid_data) < 2:
        return np.nan, np.nan
    
    corr, p_value = spearmanr(valid_data['accuracy'], valid_data['rho'])
    return corr, p_value


def create_accuracy_rho_scatter(df: pd.DataFrame, output_path: str = "accuracy_vs_rho_scatter.png"):
    """
    Create scatter plot of Accuracy vs Rho.
    """
    valid_data = df.dropna(subset=['accuracy', 'rho'])
    
    if len(valid_data) == 0:
        print("No valid data for scatter plot")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Color by architecture if available
    if 'architecture' in valid_data.columns:
        architectures = valid_data['architecture'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(architectures)))
        color_map = dict(zip(architectures, colors))
        
        for arch in architectures:
            arch_data = valid_data[valid_data['architecture'] == arch]
            plt.scatter(arch_data['accuracy'], arch_data['rho'], 
                       label=arch, alpha=0.6, s=50, c=[color_map[arch]])
    else:
        plt.scatter(valid_data['accuracy'], valid_data['rho'], alpha=0.6, s=50)
    
    plt.xlabel('Accuracy', fontsize=14, fontweight='bold')
    plt.ylabel('Ricci Curvature (Rho)', fontsize=14, fontweight='bold')
    plt.title('Accuracy vs. Ricci Curvature (Rho)', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add correlation line
    z = np.polyfit(valid_data['accuracy'], valid_data['rho'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid_data['accuracy'].min(), valid_data['accuracy'].max(), 100)
    plt.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'Trend line')
    
    # Add correlation text
    corr, p_val = spearmanr(valid_data['accuracy'], valid_data['rho'])
    plt.text(0.05, 0.95, f'Spearman ρ = {corr:.4f}\np-value = {p_val:.2e}',
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if 'architecture' in valid_data.columns:
        plt.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved scatter plot to: {output_path}")
    plt.close()


def analyze_dropout_impact(df: pd.DataFrame, output_path: str = "dropout_analysis.png"):
    """
    Analyze the impact of dropout rate on Accuracy and Rho.
    Note: Dropout rate may need to be extracted from metadata or file names.
    """
    # If dropout_rate column doesn't exist, try to extract it
    if 'dropout_rate' not in df.columns:
        print("⚠ Dropout rate information not found in data.")
        print("  Creating placeholder analysis based on architecture...")
        
        # For now, create a simplified analysis
        if 'architecture' in df.columns:
            plt.figure(figsize=(14, 6))
            
            # Plot 1: Accuracy by architecture
            plt.subplot(1, 2, 1)
            df.boxplot(column='accuracy', by='architecture', ax=plt.gca())
            plt.title('Accuracy by Architecture', fontweight='bold')
            plt.suptitle('')  # Remove default title
            plt.xlabel('Architecture')
            plt.ylabel('Accuracy')
            
            # Plot 2: Rho by architecture
            plt.subplot(1, 2, 2)
            df.boxplot(column='rho', by='architecture', ax=plt.gca())
            plt.title('Ricci Curvature (Rho) by Architecture', fontweight='bold')
            plt.suptitle('')
            plt.xlabel('Architecture')
            plt.ylabel('Rho')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved architecture analysis to: {output_path}")
            plt.close()
        return
    
    # If dropout_rate exists, create detailed analysis
    valid_data = df.dropna(subset=['dropout_rate', 'accuracy', 'rho'])
    
    if len(valid_data) == 0:
        print("No valid dropout data for analysis")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Accuracy vs Dropout
    dropout_values = sorted(valid_data['dropout_rate'].unique())
    acc_by_dropout = [valid_data[valid_data['dropout_rate'] == d]['accuracy'].values 
                      for d in dropout_values]
    
    axes[0].boxplot(acc_by_dropout, labels=dropout_values)
    axes[0].set_xlabel('Dropout Rate', fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontweight='bold')
    axes[0].set_title('Accuracy by Dropout Rate', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Rho vs Dropout
    rho_by_dropout = [valid_data[valid_data['dropout_rate'] == d]['rho'].values 
                      for d in dropout_values]
    
    axes[1].boxplot(rho_by_dropout, labels=dropout_values)
    axes[1].set_xlabel('Dropout Rate', fontweight='bold')
    axes[1].set_ylabel('Ricci Curvature (Rho)', fontweight='bold')
    axes[1].set_title('Ricci Curvature by Dropout Rate', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved dropout analysis to: {output_path}")
    plt.close()


def perform_statistical_tests(df: pd.DataFrame) -> Dict[str, float]:
    """
    Perform parametric tests to validate "better network performance = better Ricci score" hypothesis.
    
    Returns:
        Dictionary with test results
    """
    valid_data = df.dropna(subset=['accuracy', 'rho'])
    
    if len(valid_data) < 4:
        print("Not enough data for statistical tests")
        return {}
    
    results = {}
    
    # Test 1: Split data into high/low accuracy groups and compare Rho
    median_acc = valid_data['accuracy'].median()
    high_acc = valid_data[valid_data['accuracy'] >= median_acc]['rho']
    low_acc = valid_data[valid_data['accuracy'] < median_acc]['rho']
    
    if len(high_acc) > 1 and len(low_acc) > 1:
        t_stat, p_value = ttest_ind(high_acc, low_acc)
        results['t_test_high_vs_low_accuracy'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'mean_rho_high_acc': high_acc.mean(),
            'mean_rho_low_acc': low_acc.mean(),
            'n_high': len(high_acc),
            'n_low': len(low_acc)
        }
    
    # Test 2: Pearson correlation (parametric version)
    if len(valid_data) >= 2:
        pearson_r, pearson_p = pearsonr(valid_data['accuracy'], valid_data['rho'])
        results['pearson_correlation'] = {
            'correlation': pearson_r,
            'p_value': pearson_p,
            'n_samples': len(valid_data)
        }
    
    # Test 3: Top vs Bottom quartile comparison
    q75 = valid_data['accuracy'].quantile(0.75)
    q25 = valid_data['accuracy'].quantile(0.25)
    
    top_quartile = valid_data[valid_data['accuracy'] >= q75]['rho']
    bottom_quartile = valid_data[valid_data['accuracy'] <= q25]['rho']
    
    if len(top_quartile) > 1 and len(bottom_quartile) > 1:
        t_stat, p_value = ttest_ind(top_quartile, bottom_quartile)
        results['t_test_top_vs_bottom_quartile'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'mean_rho_top': top_quartile.mean(),
            'mean_rho_low': bottom_quartile.mean(),
            'n_top': len(top_quartile),
            'n_bottom': len(bottom_quartile)
        }
    
    return results


def print_analysis_summary(df: pd.DataFrame, spearman_result: Tuple[float, float], 
                          test_results: Dict):
    """Print comprehensive analysis summary."""
    print("\n" + "=" * 100)
    print("ANALYSIS SUMMARY")
    print("=" * 100)
    
    print(f"\n1. DATA OVERVIEW:")
    print(f"   Total data points: {len(df)}")
    print(f"   Valid data points: {len(df.dropna(subset=['accuracy', 'rho']))}")
    print(f"   Unique runs: {df['run_name'].nunique() if 'run_name' in df.columns else 'N/A'}")
    
    if 'architecture' in df.columns:
        print(f"\n   Architecture distribution:")
        print(df['architecture'].value_counts().to_string())
    
    valid_data = df.dropna(subset=['accuracy', 'rho'])
    print(f"\n2. DESCRIPTIVE STATISTICS:")
    print(f"   Accuracy: mean={valid_data['accuracy'].mean():.4f}, "
          f"std={valid_data['accuracy'].std():.4f}, "
          f"min={valid_data['accuracy'].min():.4f}, "
          f"max={valid_data['accuracy'].max():.4f}")
    print(f"   Rho: mean={valid_data['rho'].mean():.2f}, "
          f"std={valid_data['rho'].std():.2f}, "
          f"min={valid_data['rho'].min():.2f}, "
          f"max={valid_data['rho'].max():.2f}")
    
    print(f"\n3. SPEARMAN RANK CORRELATION:")
    corr, p_val = spearman_result
    print(f"   Spearman ρ = {corr:.6f}")
    print(f"   p-value = {p_val:.2e}")
    if p_val < 0.05:
        significance = "✓ Statistically significant"
    else:
        significance = "✗ Not statistically significant"
    print(f"   {significance}")
    
    print(f"\n4. STATISTICAL TESTS:")
    if 'pearson_correlation' in test_results:
        pearson = test_results['pearson_correlation']
        print(f"   Pearson Correlation:")
        print(f"     r = {pearson['correlation']:.6f}, p = {pearson['p_value']:.2e}")
    
    if 't_test_high_vs_low_accuracy' in test_results:
        ttest = test_results['t_test_high_vs_low_accuracy']
        print(f"   T-test (High vs Low Accuracy):")
        print(f"     High accuracy group mean Rho = {ttest['mean_rho_high_acc']:.2f} (n={ttest['n_high']})")
        print(f"     Low accuracy group mean Rho = {ttest['mean_rho_low_acc']:.2f} (n={ttest['n_low']})")
        print(f"     t-statistic = {ttest['t_statistic']:.4f}, p = {ttest['p_value']:.2e}")
        if ttest['p_value'] < 0.05:
            print(f"     ✓ Significant difference")
        else:
            print(f"     ✗ No significant difference")
    
    print("\n" + "=" * 100 + "\n")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main analysis pipeline."""
    # Configuration
    BASE_DIR = os.path.join(os.getcwd(), 'output_layers')
    K_VALUE = None  # None means use k from master summary, or specify a fixed value like 100
    OUTPUT_DIR = os.getcwd()
    
    print("\n" + "=" * 100)
    print("DEEP LEARNING AS RICCI FLOW - COMPREHENSIVE ANALYSIS")
    print("=" * 100)
    print(f"Base directory: {BASE_DIR}")
    if K_VALUE is None:
        print(f"k value: Auto-detect from master summary")
    else:
        print(f"k value for kNN graphs: {K_VALUE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 100 + "\n")
    
    # Step 1: Collect all data
    df = collect_all_runs_data(BASE_DIR, k=K_VALUE)
    
    if len(df) == 0:
        print("ERROR: No data collected. Please check the output_layers/ directory.")
        return
    
    # Save collected data
    output_csv = os.path.join(OUTPUT_DIR, "ricci_analysis_data.csv")
    df.to_csv(output_csv, index=False)
    print(f"Saved collected data to: {output_csv}\n")
    
    # Step 2: Calculate Spearman correlation
    print("Calculating Spearman Rank Correlation...")
    spearman_result = calculate_spearman_correlation(df)
    corr, p_val = spearman_result
    print(f"Spearman ρ = {corr:.6f}, p-value = {p_val:.2e}\n")
    
    # Step 3: Create visualizations
    print("Creating visualizations...")
    create_accuracy_rho_scatter(df, os.path.join(OUTPUT_DIR, "accuracy_vs_rho_scatter.png"))
    analyze_dropout_impact(df, os.path.join(OUTPUT_DIR, "dropout_analysis.png"))
    
    # Step 4: Perform statistical tests
    print("Performing statistical tests...")
    test_results = perform_statistical_tests(df)
    
    # Step 5: Print summary
    print_analysis_summary(df, spearman_result, test_results)
    
    # Save test results
    if test_results:
        results_df = pd.DataFrame([test_results]).T
        results_df.columns = ['value']
        results_df.to_csv(os.path.join(OUTPUT_DIR, "statistical_test_results.csv"))
        print(f"Saved statistical test results to: statistical_test_results.csv")
    
    print("Analysis complete! ✓")


if __name__ == "__main__":
    main()

