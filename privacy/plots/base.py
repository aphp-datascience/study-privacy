from typing import List, Tuple

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches as mpatches
from pandas import DataFrame

from privacy.plots.utils import *

sns.set_theme(style="whitegrid")


def scatter_plot(
    df: DataFrame,
    x: str,
    y: str,
    hue: str,
    style: str,
    xlabel: str,
    ylabel: str,
    figsize: Tuple[int, int] = (8, 8),
    palette: List[str] = palette,
    x_log_scale: bool = False,
    y_log_scale: bool = True,
    rotate_x_ticks: bool = False,
    conf_name: Optional[str] = None,
    file_name: Optional[str] = None,
):
    # Initialize figure
    fig, ax = plt.subplots(
        figsize=figsize,
    )

    # Scatterplot
    sns.scatterplot(
        ax=ax,
        data=df,
        x=x,
        y=y,
        hue=hue,
        style=style,
        s=330,
        palette=palette,
    )

    # Set axe labels
    ax.set(xlabel=xlabel, ylabel=ylabel)

    # Set legend
    handles, labels = ax.get_legend_handles_labels()
    hue_labels = [legend_rename(key) for key in labels]

    legend = ax.legend(
        handles=handles,
        labels=hue_labels,
        fontsize="small",
        loc="upper right",
        bbox_to_anchor=(1.43, 1),
        title=None,
        frameon=True,
    )

    # Sex X axe to log scale
    if x_log_scale:
        ax.set(
            xscale="log",
        )
    if y_log_scale:
        ax.set(
            yscale="log",
        )
    else:
        # Set y lim to 0
        ax.set_ylim(0)

    # Rotate x ticks
    if rotate_x_ticks:
        plt.xticks(rotation=90)

    # Columns
    cols = [x, y, hue, style]

    # Show or save plot
    show_or_save(
        fig,
        filename=file_name,
        conf_name=conf_name,
        legend=[
            legend,
        ],
        tables=[
            df[cols],
        ],
    )


def bar_plot(
    df: DataFrame,
    y: str,
    y_label: str,
    x_label: str,
    conf_name: Optional[str] = None,
    file_name: Optional[str] = None,
):
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1 = sns.barplot(x=df.index, y=df[y], color="#257B59", alpha=0.5, order=df.index)

    ticks = ax1.get_xticks()
    labels = df.index
    ax1.set_xticks(ticks, labels, rotation=90)

    ax1.grid(False, axis="x")

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)

    red_patch = mpatches.Patch(color="#257B59", label=y_label, alpha=0.5)

    legend = ax1.legend(
        handles=[red_patch],
        loc="best",
    )
    ax1.ticklabel_format(axis="y", style="plain")

    # Show or save plot
    show_or_save(
        fig,
        filename=file_name,
        conf_name=conf_name,
        legend=[
            legend,
        ],
        tables=[
            df,
        ],
        **{"save_index": True},
    )


def bar_scatter_plot(
    df,
    y_bar,
    y_scatter,
    y_bar_label,
    y_scatter_label,
    x_label,
    y_log_scale: bool = False,
    conf_name: Optional[str] = None,
    file_name: Optional[str] = None,
):
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(15, 5))
    ax1 = sns.barplot(
        x=df.index, y=df[y_bar], color="#257B59", alpha=0.5, order=df.index
    )

    ticks = ax1.get_xticks()
    labels = df.index
    ax1.set_xticks(ticks, labels, rotation=90)
    ax1.ticklabel_format(axis="y", style="plain")

    ax2 = ax1.twinx()

    ax2 = sns.scatterplot(x=df.index, y=df[y_scatter], color="black")
    ax1.grid(False)
    ax2.grid(False)
    ax2.set_ylim(0, 1)

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_bar_label)
    ax2.set_ylabel(y_scatter_label)

    if y_log_scale:
        ax1.set(
            yscale="log",
        )

    patch_bars = mpatches.Patch(color="#257B59", label=y_bar_label, alpha=0.5)
    patch_points = mlines.Line2D(
        [],
        [],
        color="black",
        marker=".",
        linestyle="None",
        markersize=10,
        label=y_scatter_label,
    )

    legend = ax1.legend(
        handles=[patch_points, patch_bars],
        bbox_to_anchor=(0.17, -0.25),
    )

    # Show or save plot
    show_or_save(
        fig,
        filename=file_name,
        conf_name=conf_name,
        legend=[
            legend,
        ],
        tables=[
            df,
        ],
        **{"save_index": True},
    )
