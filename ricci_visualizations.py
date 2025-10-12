"""Utility functions for visualising results from Ricci‑flow experiments.

This module provides a collection of functions to load the CSV summaries
produced by the ricci‑flow training scripts and to generate a suite of plots
that parallel those shown in the accompanying paper.  All plots are built
using `matplotlib` only to remain self contained and avoid reliance on
external plotting libraries.  The functions are designed to be imported
into an interactive notebook; they will show their figures immediately.

Example usage:

    import ricci_visualizations as rv
    df = rv.load_runs_summary('runs_summary.csv')
    # Scatter of Ricci coefficient vs test accuracy coloured by dataset
    rv.plot_rho_vs_accuracy(df)
    # Heatmap of mean rho per width and depth for a single dataset
    rv.plot_metric_heatmap(df[df['dataset']=="mnist_1v7"], metric='rho')
    # Distribution of best k values by dataset
    rv.plot_k_distribution(df)

All plotting functions accept a pandas DataFrame as their first argument.
If you wish to save figures instead of displaying them, call
``plt.savefig('filename.png')`` after calling a plotting function but
before ``plt.show()``; alternatively set `show=False` on supported
functions.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Iterable, List, Optional, Union, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__all__ = [
    "load_runs_summary",
    "plot_rho_vs_accuracy",
    "plot_metric_heatmap",
    "plot_k_distribution",
    "plot_accuracy_heatmap",
    "plot_rho_heatmap",
    "plot_rho_vs_k_scatter",
]


def load_runs_summary(csv_path: Union[str, Path] = "runs_summary.csv") -> pd.DataFrame:
    """Load a CSV file produced by the training config sweep.

    The expected columns include: ``dataset``, ``width``, ``depth``, ``activation``,
    ``batchnorm``, ``dropout``, ``residual``, ``optimizer``, ``lr``, ``loss``,
    ``test_acc``, ``best_k``, ``rho``, ``z``, ``connected_all_layers``,
    ``elapsed_s``, ``run_id``, ``error`` and ``session``.  Only columns
    present in the file are returned – additional columns are ignored.

    Parameters
    ----------
    csv_path: str or Path
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded DataFrame.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def _setup_axis(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    """Apply consistent styling to an axis."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_rho_vs_accuracy(
    df: pd.DataFrame,
    hue: str = "dataset",
    datasets: Optional[Iterable[str]] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (6, 4),
    alpha: float = 0.7,
    markersize: float = 20.0,
    show: bool = True,
) -> plt.Axes:
    """Scatter plot of Ricci coefficient (rho) against test accuracy.

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame containing at least the columns ``rho`` and ``test_acc``.
    hue: str, optional
        Column name to colour the points by.  Default is ``dataset``.  If
        ``None`` no grouping is applied and all points are drawn with the
        same colour.
    datasets: iterable of str, optional
        If provided, only rows whose ``dataset`` value is in this list will
        be plotted.  This is useful for focusing on a subset of datasets.
    ax: matplotlib.axes.Axes, optional
        An existing axis to draw the plot on.  If ``None``, a new figure
        and axis are created with ``figsize``.
    figsize: tuple
        Size of the figure if a new figure is created.
    alpha: float
        Transparency of the markers (0–1).  Lower values make points more
        transparent, which helps with overplotting.
    markersize: float
        Size of the markers.
    show: bool
        Whether to call ``plt.show()``.  Set to ``False`` if you wish to
        further customise the figure or save it programmatically.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the scatter plot.
    """
    if datasets is not None:
        df = df[df["dataset"].isin(datasets)]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    # Grouping and plotting
    if hue is None:
        ax.scatter(df["rho"], df["test_acc"], s=markersize, alpha=alpha)
    else:
        # Determine unique values and cycle through colours
        groups = df.groupby(hue)
        for key, subdf in groups:
            ax.scatter(subdf["rho"], subdf["test_acc"], s=markersize, alpha=alpha, label=str(key))
        ax.legend(title=hue, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.5)
    _setup_axis(ax, xlabel="Ricci coefficient (ρ)", ylabel="Test accuracy", title="ρ versus accuracy")
    if show:
        plt.tight_layout()
        plt.show()
    return ax


def plot_metric_heatmap(
    df: pd.DataFrame,
    metric: str = "rho",
    index: str = "width",
    columns: str = "depth",
    aggfunc: Callable = np.mean,
    cmap: str = "coolwarm",
    annot: bool = True,
    fmt: str = ".2f",
    figsize: tuple = (6, 4),
    show: bool = True,
) -> plt.Axes:
    """Plot a heatmap of a metric aggregated over two categorical axes.

    The DataFrame must contain at least the columns named by ``index``,
    ``columns`` and ``metric``.  It is pivoted using ``index`` as the rows and
    ``columns`` as the columns, applying ``aggfunc`` when multiple rows map
    to the same cell.  NaN values are left blank.

    Parameters
    ----------
    df: pandas.DataFrame
        Input data.
    metric: str
        Name of the column containing the metric to visualise (e.g. ``rho`` or
        ``test_acc``).
    index: str
        Column to use as the heatmap's row index (e.g. ``width``).
    columns: str
        Column to use as the heatmap's column index (e.g. ``depth``).
    aggfunc: callable
        Function to aggregate values when more than one row falls into a
        single heatmap cell.  Typical choices are ``np.mean`` or ``np.median``.
    cmap: str
        Matplotlib colour map name.
    annot: bool
        Whether to annotate each cell with its value.
    fmt: str
        Format string for annotations (e.g. ".2f").
    figsize: tuple
        Size of the figure.
    show: bool
        If ``True`` the plot is shown immediately.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the heatmap.
    """
    if metric not in df.columns:
        raise KeyError(f"Column '{metric}' not found in DataFrame")
    pivot = df.pivot_table(index=index, columns=columns, values=metric, aggfunc=aggfunc)
    fig, ax = plt.subplots(figsize=figsize)
    c = ax.imshow(pivot.values, aspect="auto", cmap=cmap, origin="upper")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    _setup_axis(ax, xlabel=columns, ylabel=index, title=f"Heatmap of {metric}")
    # Annotate cells
    if annot:
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iat[i, j]
                if pd.notna(val):
                    ax.text(j, i, format(val, fmt), ha="center", va="center", color="black")
    fig.colorbar(c, ax=ax, label=metric)
    plt.tight_layout()
    if show:
        plt.show()
    return ax


def plot_k_distribution(
    df: pd.DataFrame,
    by: str = "dataset",
    bins: Optional[Iterable[int]] = None,
    figsize: tuple = (6, 4),
    show: bool = True,
) -> plt.Axes:
    """Plot histograms of the ``best_k`` values grouped by a given column.

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame containing the column ``best_k``.
    by: str
        Column to group the data by when plotting separate histograms.
    bins: iterable of int, optional
        Bin edges to use for the histogram.  If ``None`` bins are determined
        automatically using ``numpy.histogram_bin_edges``.
    figsize: tuple
        Size of the figure.
    show: bool
        Whether to display the plot.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the histogram(s).
    """
    if "best_k" not in df.columns:
        raise KeyError("Column 'best_k' not found in DataFrame")
    fig, ax = plt.subplots(figsize=figsize)
    groups = df.groupby(by)
    for key, sub in groups:
        data = sub["best_k"].dropna()
        # Determine bins if not provided
        if bins is None:
            # Use Freedman–Diaconis rule for bin width
            q25, q75 = np.percentile(data, [25, 75])
            iqr = q75 - q25
            if iqr > 0:
                bin_width = 2 * iqr / (len(data) ** (1 / 3))
                bins_count = max(1, int(np.ceil((data.max() - data.min()) / bin_width)))
                bin_edges = np.linspace(data.min(), data.max(), bins_count + 1)
            else:
                bin_edges = np.linspace(data.min(), data.max(), 10)
        else:
            bin_edges = bins
        ax.hist(data, bins=bin_edges, alpha=0.6, label=str(key), edgecolor="black")
    ax.legend(title=by, bbox_to_anchor=(1.05, 1), loc="upper left")
    _setup_axis(ax, xlabel="best_k", ylabel="count", title="Distribution of best_k")
    plt.tight_layout()
    if show:
        plt.show()
    return ax


def plot_accuracy_heatmap(
    df: pd.DataFrame,
    index: str = "width",
    columns: str = "depth",
    aggfunc: Callable = np.mean,
    figsize: tuple = (6, 4),
    show: bool = True,
) -> plt.Axes:
    """Heatmap of test accuracy averaged over specified axes.

    Simply wraps ``plot_metric_heatmap`` with ``metric='test_acc'``.
    """
    return plot_metric_heatmap(df, metric="test_acc", index=index, columns=columns, aggfunc=aggfunc, figsize=figsize, show=show)


def plot_rho_heatmap(
    df: pd.DataFrame,
    index: str = "width",
    columns: str = "depth",
    aggfunc: Callable = np.mean,
    figsize: tuple = (6, 4),
    show: bool = True,
) -> plt.Axes:
    """Heatmap of Ricci coefficient averaged over specified axes.

    Simply wraps ``plot_metric_heatmap`` with ``metric='rho'``.
    """
    return plot_metric_heatmap(df, metric="rho", index=index, columns=columns, aggfunc=aggfunc, figsize=figsize, show=show)


def plot_rho_vs_k_scatter(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    hue: Optional[str] = "dataset",
    figsize: tuple = (6, 4),
    alpha: float = 0.7,
    markersize: float = 20.0,
    show: bool = True,
) -> plt.Axes:
    """Scatter of Ricci coefficient against the selected number of neighbours (best_k).

    Useful to assess how the optimal k relates to the computed Ricci values.

    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame containing ``best_k`` and ``rho``.
    ax: matplotlib.axes.Axes, optional
        Axis to draw on; if ``None`` a new axis is created.
    hue: str or None
        Column to group points by for colouring.  If ``None`` no grouping is applied.
    figsize: tuple
        Size of the figure if a new axis is created.
    alpha: float
        Transparency of the markers.
    markersize: float
        Size of the markers.
    show: bool
        Whether to show the figure immediately.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the scatter plot.
    """
    if "best_k" not in df.columns or "rho" not in df.columns:
        raise KeyError("DataFrame must contain 'best_k' and 'rho' columns")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if hue is None:
        ax.scatter(df["best_k"], df["rho"], s=markersize, alpha=alpha)
    else:
        for key, sub in df.groupby(hue):
            ax.scatter(sub["best_k"], sub["rho"], s=markersize, alpha=alpha, label=str(key))
        ax.legend(title=hue, bbox_to_anchor=(1.05, 1), loc="upper left")
    _setup_axis(ax, xlabel="best_k (optimal k)", ylabel="Ricci coefficient (ρ)", title="ρ versus k")
    plt.tight_layout()
    if show:
        plt.show()
    return ax