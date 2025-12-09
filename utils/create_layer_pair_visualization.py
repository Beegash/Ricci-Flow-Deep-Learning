import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib.colors import LinearSegmentedColormap

def create_layer_pair_visualization(msc_path, mfr_path, output_path):
    """
    Create a scatter plot of total curvature vs total geodesic change,
    with one point per layer pair, matching the style of the reference image.
    """
    # Read the CSV files
    msc = pd.read_csv(msc_path)
    mfr = pd.read_csv(mfr_path)
    
    # Align mfr with msc: msc layer l corresponds to mfr layer l-1
    mfr_shifted = mfr.copy()
    mfr_shifted['layer'] = mfr_shifted['layer'] + 1
    
    # Merge the dataframes
    merged = msc.merge(mfr_shifted, on=['mod', 'layer'], suffixes=('_dg', '_fr'))
    
    # Calculate r_all correlation (on ALL individual data points, including layer 1)
    # This matches the r_all from correlation_report function
    r_all, p_all = pearsonr(merged["ssr_dg"].values, merged["ssr_fr"].values)
    
    # Filter out layer 1 for visualization (since it doesn't have a proper layer pair)
    merged_vis = merged[merged['layer'] != 1]
    
    # ========================================================================
    # WHY AGGREGATION IS NEEDED:
    # ========================================================================
    # The raw data has 700 individual points (70 models × 10 layer pairs)
    # But the requirement is: "the number of the dots should be layer-1" 
    # This means we need exactly 10 dots (one per layer pair: 2-1, 3-2, ..., 11-10)
    # 
    # Without aggregation: 700 points (too many, doesn't match requirement)
    # With aggregation: 10 points (one per layer pair, matches requirement)
    #
    # The aggregation sums values across all models for each layer pair,
    # giving us one representative point per layer pair for visualization.
    # ========================================================================
    
    # Aggregate per layer pair: sum across all models for each layer
    # This gives one point per layer pair for visualization (10 points total)
    # Use merged_vis (excluding layer 1) for the plot
    aggregated = merged_vis.groupby('layer').agg({
        'ssr_dg': 'sum',  # Total geodesic change (summed across all 70 models)
        'ssr_fr': 'sum'   # Total curvature (summed across all 70 models)
    }).reset_index()
    
    # Sort by layer
    aggregated = aggregated.sort_values('layer')
    
    # Calculate Pearson correlation on aggregated data
    # This matches the paper's methodology (Equation 8): correlation over L-1 data points
    correlation_aggregated, p_value_aggregated = pearsonr(aggregated['ssr_dg'], aggregated['ssr_fr'])
    
    # Use aggregated correlation for the plot (matching the paper's method)
    # The paper computes correlation over L-1 aggregated values (one per layer pair)
    correlation = correlation_aggregated
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a custom colormap from light yellow to dark blue
    # Matching the image: lightest yellow (layer 2-1) to darkest blue (layer 11-10)
    from matplotlib.colors import LinearSegmentedColormap
    
    # Define colors from light yellow to dark blue
    colors_list = ['#FFFFE0', '#FFFF99', '#FFFF00', '#CCFF00', '#99FF00', 
                   '#66FF00', '#33FF00', '#00CCFF', '#0099FF', '#0066FF', '#0033FF']
    
    # Create a colormap
    n_layers = len(aggregated)
    layer_min = aggregated['layer'].min()
    layer_max = aggregated['layer'].max()
    
    # Map each layer to a color index
    layer_colors = {}
    for idx, layer in enumerate(sorted(aggregated['layer'].unique())):
        # Use colors from yellow (light) to blue (dark)
        color_idx = int((idx / (n_layers - 1)) * (len(colors_list) - 1)) if n_layers > 1 else 0
        layer_colors[int(layer)] = colors_list[color_idx]
    
    # Scatter plot with color coding by layer pair
    legend_elements = []
    for idx, row in aggregated.iterrows():
        layer = int(row['layer'])
        color = layer_colors[layer]
        ax.scatter(row['ssr_dg'], row['ssr_fr'], 
                  c=color, 
                  s=200, 
                  edgecolors='black',
                  linewidth=1.5,
                  alpha=0.9,
                  zorder=3)
        # Create legend element
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                          markerfacecolor=color, 
                                          markersize=12, 
                                          markeredgecolor='black',
                                          markeredgewidth=1.5,
                                          label=f'layer {layer}-layer {layer-1}'))
    
    # Add regression line
    if len(aggregated) >= 2:
        z = np.polyfit(aggregated['ssr_dg'], aggregated['ssr_fr'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(aggregated['ssr_dg'].min(), aggregated['ssr_dg'].max(), 100)
        ax.plot(x_line, p(x_line), 'k-', linewidth=2.5, alpha=0.9, zorder=2)
    
    # Add correlation coefficient text (using rho symbol) in upper right corner
    ax.text(0.95, 0.95, f'ρ = {correlation:.2f}', 
            transform=ax.transAxes, 
            fontsize=16, 
            verticalalignment='top',
            horizontalalignment='right',
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
    
    # Labels
    ax.set_xlabel('total geodesic change', fontsize=14, fontweight='bold')
    ax.set_ylabel('total curvature', fontsize=14, fontweight='bold')
    
    # Format axes with scientific notation if needed
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0), useMathText=True)
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # Grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
    
    # Create legend on the right side
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_path}")
    print(f"Aggregated correlation (matching paper's method, L-1 points): {correlation_aggregated:.4f}")
    print(f"r_all correlation (on all individual points, for comparison): {r_all:.4f}")
    print(f"Number of layer pairs shown in plot (aggregated points): {len(aggregated)}")
    print(f"Total number of individual data points: {len(merged)}")
    print(f"Note: Using aggregated correlation to match paper's methodology (Equation 8)")

if __name__ == "__main__":
    import sys
    
    # Allow command line argument or use default
    if len(sys.argv) > 1:
        # If CSV path provided, extract base directory
        csv_path = sys.argv[1]
        base_dir = csv_path.rsplit('/', 1)[0]  # Get directory containing the CSV
    else:
        # Default: synthetic_c model
        base_dir = "/Users/cihan/Documents/GitHub/Ricci-Flow-Deep-Learning---Graduation-Project/output_layers/wide_11_synthetic_c/analysis_k100"
    
    msc_path = f"{base_dir}/msc.csv"
    mfr_path = f"{base_dir}/mfr.csv"
    output_path = f"{base_dir}/layer_pair_visualization.png"
    
    create_layer_pair_visualization(msc_path, mfr_path, output_path)

