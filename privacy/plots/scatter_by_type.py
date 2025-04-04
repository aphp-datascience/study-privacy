from typing import Dict, Optional

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import patches as mpatches
from matplotlib.font_manager import FontProperties

from privacy.misc.constants import (
    cohort_name_mapping,
    colors_cohorts,
    colors_cohorts_inverse_mapping,
)
from privacy.plots.utils import markers, show_or_save


def scatter_by_type(
    df: pd.DataFrame,
    alpha_dict: Dict[int, int] = {100: 0.3, 1000: 1},
    x: str = "cohort_name",
    y: str = "uniqueness",
    xlabel: str = "Cohort",
    ylabel: str = "Uniqueness",
    style: str = "pseudonymization_algorithm",
    h: int = 8,
    color: Optional[str] = "black",
    hue: Optional[str] = None,
    palette: Optional[Dict[str, str]] = None,
    add_cohort_to_legend: bool = True,
    conf_name: Optional[str] = None,
    file_name: Optional[str] = None,
):
    # Initialize figure
    figsize = (h, h)
    fig, ax = plt.subplots(
        figsize=figsize,
    )

    # First plot all combinations of NoPseudonymizer
    for key, value in alpha_dict.items():
        ax = sns.scatterplot(
            ax=ax,
            data=df.loc[(df.high_general == key)],
            x=x,
            y=y,
            style=style,
            s=330,
            markers=markers,
            alpha=value,
            color=color,
            legend=True,
            hue=hue,
            palette=palette,
        )

    # Plot NoPseudonymizer for each cohort
    for cohort, color in colors_cohorts_inverse_mapping.items():
        ax = sns.scatterplot(
            ax=ax,
            # data=df2,
            data=df.loc[
                (
                    df.pseudonymization_algorithm.isin(
                        ["NoPseudonymizer", "No pseudonymisation"]
                    )
                )
                & (df.cohort_name == cohort)
            ],
            x=x,
            y=y,
            style="pseudonymization_algorithm",
            s=330,
            markers=markers,
            ec=color,
            fc="none",
            legend=True,
        )

    # Set labels
    ax.set(xlabel=xlabel, ylabel=ylabel)

    # Set y lim to 0
    ax.set_ylim(0)

    # Rotate ticks
    plt.xticks(rotation=90, fontsize=13)

    # Build legend (cohorts)
    cohort_patches = []
    for cohort, color in colors_cohorts.items():
        patch = mpatches.Patch(color=color, label=cohort_name_mapping[cohort], alpha=1)
        cohort_patches.append(patch)

    title_cohorts = [
        mpatches.Patch(
            color=None,
            label="Cohort",
            fill=False,
        ),
    ]

    # Build legend (shift dates)
    alpha_patches = []
    for shift, alpha in alpha_dict.items():
        patch = mpatches.Patch(
            color="black",
            label=shift,
            alpha=alpha,
        )

        patch_points = mlines.Line2D(
            [],
            [],
            color="black",
            marker=".",
            linestyle="None",
            markersize=15,
            label=shift,
            alpha=alpha,
        )
        alpha_patches.append(patch_points)

    title_shift_dates = [
        mpatches.Patch(
            color=None,
            label=r"Shift Parameter (days)",
            fill=False,
        ),
    ]

    # Build legend (pseudonymization algorithm)
    handles, labels = ax.get_legend_handles_labels()
    dict_handle_labels = {}
    for handle, label in zip(handles, labels):
        handle.set_facecolor("none")
        if label in [
            "BasePseudonymizer",
            "NoPseudonymizer",
            "BirthPseudonymizer",
            "StayPseudonymizer",
            "No pseudonymisation",
            "Base pseudonymisation",
            "Birth pseudonymisation",
            "Hospital stay pseudonymisation",
        ]:
            dict_handle_labels.update({label: handle})

    title_pseudo_algo = [
        mpatches.Patch(
            color=None,
            label=r"Pseudonymisation Scheme",
            fill=False,
        ),
    ]

    # Legend handles
    if add_cohort_to_legend:
        final_handles = (
            title_pseudo_algo
            + list(dict_handle_labels.values())
            + title_cohorts
            + cohort_patches
            + title_shift_dates
            + alpha_patches
        )
    else:
        final_handles = (
            title_pseudo_algo
            + list(dict_handle_labels.values())
            + title_shift_dates
            + alpha_patches
        )

    # Build legend
    legend = plt.legend(
        handles=final_handles,
        ncol=1,
        bbox_to_anchor=(1, 1),
        loc="best",
        title=None,
        frameon=True,
    )
    font_properties = FontProperties(weight="bold", size=12)

    for text in legend.get_texts():
        if text.get_text() in [
            "Shift Parameter (days)",
            "Pseudonymisation Scheme",
            "Cohort",
        ]:
            text.set_fontproperties(font_properties)

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
