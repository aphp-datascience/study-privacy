from typing import Optional

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches as mpatches
from pandas import DataFrame, IndexSlice

from privacy.plots.utils import *


def population_pyramid(
    df: DataFrame,
    title: str = "Female | Male   ",
    y_bar: str = "count",
    x_bar_label: str = "Number of patients",
    y_label: str = "Age at first hospitalisation",
    y_scatter: str = "uniqueness",
    x_scatter_label: str = "Uniqueness",
    bar_color: str = "#257B59",
    conf_name: Optional[str] = None,
    file_name: Optional[str] = None,
):
    idxs = IndexSlice
    result_age_female = df.loc[idxs[:, "f"], :]
    result_age_male = df.loc[idxs[:, "m"], :]

    # Index
    age_cats = result_age_male.index.get_level_values(0)

    # Initiate figure
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Axis for men
    ax1 = sns.barplot(
        x=result_age_male[y_bar],
        y=age_cats,
        order=age_cats.categories,
        color=bar_color,
        alpha=0.5,
    )

    # Axis for women
    ax2 = sns.barplot(
        x=-result_age_female[y_bar],
        y=age_cats,
        order=age_cats.categories,
        color=bar_color,
        alpha=0.5,
    )

    # Instantiate an axis for the scatter plot
    ax3 = ax1.twiny()  # instantiate a second axes that shares the same y-axis

    # Scatter plot for the men
    ax3 = sns.scatterplot(x=result_age_male[y_scatter], y=age_cats, color="black")
    # Scatter plot for the women
    ax4 = sns.scatterplot(x=-result_age_female[y_scatter], y=age_cats, color="black")

    ### Aesthetics ###
    # Set x lim
    max_value = df[y_bar].max()
    ax1.set_xlim(-max_value, max_value)
    ax3.set_xlim(-1, 1)

    # Ticks
    ticks = ax1.get_xticks()
    labels = [str(int(abs(t))) for t in ticks]
    ax1.set_xticks(ticks, labels, rotation=90)

    ticks = ax3.get_xticks()
    labels = [str((abs(t))) for t in ticks]
    ax3.set_xticks(ticks, labels, rotation=45)

    # Labels
    ax1.set_xlabel(x_bar_label)
    ax1.set_ylabel(y_label)

    # Vertical line at 0
    ax1.axvline(0, color="black")
    ax3.set_xlabel(x_scatter_label)

    # Legend
    bar_patch = mpatches.Patch(color=bar_color, label=x_bar_label, alpha=0.5)
    point_patch = mlines.Line2D(
        [],
        [],
        color="black",
        marker=".",
        linestyle="None",
        markersize=10,
        label=x_scatter_label,
    )

    legend = ax1.legend(
        handles=[point_patch, bar_patch],
        bbox_to_anchor=(0.17, -0.25),
    )
    plt.title(title, pad=10.2, fontweight="bold")
    plt.grid()

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
