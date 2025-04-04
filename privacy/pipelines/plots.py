from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from confit import Config
from matplotlib import patches as mpatches
from matplotlib.font_manager import FontProperties

from privacy.misc.constants import (
    cohort_name_mapping,
    colors_cohorts,
    colors_cohorts_inverse_mapping,
)
from privacy.pipelines.reliability import get_pseudonimized_dataset
from privacy.pipelines.table_n_patients_sr import get_table_n_patients_sr
from privacy.pipelines.table_cohort_uniqueness import get_table_cohort_uniqueness
from privacy.plots.base import bar_plot, bar_scatter_plot, scatter_plot
from privacy.plots.countour import countour_subplot, get_x_y_z, uniqueness_dt_subplot
from privacy.plots.population_age_pyramid import population_pyramid
from privacy.plots.scatter_by_type import scatter_by_type
from privacy.plots.utils import markers, show_or_save
from privacy.registry import registry
from privacy.misc.constants import pseudonimizer_name_mapping

sns.set_theme(style="whitegrid")


class Obj:
    pass


@registry.plots("n_patients_sr")
def n_patients_sr(
    conf: Dict[str, Any],
    conf_name: Optional[str] = None,
    file_name: Optional[str] = "n_patients_sr",
    alpha_dict: Dict[int, int] = {100: 0.3, 1000: 1},
    colors_cohorts: Dict[str, str] = colors_cohorts,
    **kwargs,
):
    df = get_table_n_patients_sr(conf)

    # Set alpha (visualization)
    for key, value in alpha_dict.items():
        df.loc[df.high_general == key, "alpha"] = value
    df["alpha"].fillna(0, inplace=True)

    # Initialize figure
    h = 6
    figsize = (h, np.sqrt(2) * h)
    fig, ax = plt.subplots(
        figsize=figsize,
    )

    # First plot all combinations of NoPseudonymizer
    for alpha in alpha_dict.values():
        for cohort, color in colors_cohorts.items():
            ax = sns.scatterplot(
                ax=ax,
                data=df.loc[
                    (df.alpha == alpha)
                    & (df.cohort_name == cohort)
                    & (df.pseudonymization_algorithm != "NoPseudonymizer")
                    & (df.pseudonymization_algorithm != "No pseudonymisation")
                ],
                x="n",
                y="success_rate",
                style="pseudonymization_algorithm",
                s=330,
                markers=markers,
                alpha=alpha,
                color=color,
                legend=True,
            )

    # Plot NoPseudonymizer for each cohort
    for cohort, color in colors_cohorts.items():
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
            x="n",
            y="success_rate",
            style="pseudonymization_algorithm",
            s=330,
            markers=markers,
            ec=color,
            fc="none",
            legend=True,
        )

    # Set labels
    ax.set(xlabel="No. patients in the accessed cohort", ylabel="P (Success)")

    # Set log scale

    ax.set(xscale="log", yscale="log")

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
        dict_handle_labels.update({label: handle})

    title_pseudo_algo = [
        mpatches.Patch(
            color=None,
            label=r"Pseudonymisation Scheme",
            fill=False,
        ),
    ]

    # Legend handles
    final_handles = (
        title_pseudo_algo
        + list(dict_handle_labels.values())
        + title_cohorts
        + cohort_patches
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

    return Obj()


@registry.plots("cohort_uniqueness")
def cohort_uniqueness(
    conf: Dict[str, Any],
    conf_name: Optional[str] = None,
    file_name: Optional[str] = "cohort_uniqueness",
    alpha_dict: Dict[int, int] = {100: 0.3, 1000: 1},
    **kwargs,
):
    df = get_table_cohort_uniqueness(conf)
    x = scatter_by_type(
        df=df,
        conf_name=conf_name,
        file_name=file_name,
        alpha_dict=alpha_dict,
        color=None,
        hue="cohort_name",
        palette=colors_cohorts_inverse_mapping,
    )

    return Obj()


@registry.plots("knowledge_uniqueness")
def knowledge_uniqueness(
    conf: Dict[str, Any],
    conf_name: Optional[str] = None,
    file_name: Optional[str] = "knowledge_uniqueness",
    alpha_dict: Dict[int, int] = {10: 0.3, 100: 0.65, 1000: 1},
    **kwargs,
):
    table3c = pd.read_csv(conf["table_knowledge_uniqueness"]["output_path"])
    table3c.cohort_name.replace(cohort_name_mapping, inplace=True)
    table3c["pseudonymization_algorithm"].replace(
        pseudonimizer_name_mapping, inplace=True
    )

    scatter_by_type(
        df=table3c,
        x="attack_knowledge",
        xlabel="Attacker's à priori knowledge",
        conf_name=conf_name,
        file_name=file_name,
        alpha_dict=alpha_dict,
        add_cohort_to_legend=False,
    )

    return Obj()


@registry.plots("simultaneous_variations_uniqueness")
def simultaneous_variations_uniqueness(
    conf: Dict[str, Any],
    conf_name: Optional[str] = None,
    file_name: Optional[str] = "simultaneous_variations_uniqueness",
    log_scale: bool = False,
    **kwargs,
):
    df = pd.read_csv(conf["table_simultaneous_variations_uniqueness"]["output_path"])
    max_value_plot = str(conf["table_simultaneous_variations_uniqueness"]["max_value_plot"])

    # Pivot values
    df_pivot = df.query(
        f"high_general<={max_value_plot} & high_birth_date<={max_value_plot}"
    ).pivot(index="high_general", columns="high_birth_date", values="uniqueness")

    x, y, z = get_x_y_z(df_pivot=df_pivot)

    # Create plot
    fig, axes = plt.subplots(
        figsize=(10, 17),
        nrows=3,
        ncols=1,
    )

    # Countour subplot
    countour_subplot(x=x, y=y, z=z, ax=axes[0], log_scale=log_scale)

    # Line plot 1
    uniqueness_dt_subplot(ax=axes[1], df=df, x="high_birth_date", hue="high_general")

    # Line plot 2
    uniqueness_dt_subplot(
        ax=axes[2],
        df=df,
        x="high_general",
        hue="high_birth_date",
    )

    # Show or save plot
    show_or_save(
        fig,
        filename=file_name,
        conf_name=conf_name,
        tables=[
            df_pivot,
            df[
                [
                    "high_general",
                    "high_birth_date",
                    "uniqueness",
                ]
            ],
        ],
        **{"save_index": True},
    )

    return Obj()


@registry.plots("age_pyramid_uniqueness")
def age_pyramid_uniqueness(
    conf: Dict[str, Any],
    conf_name: Optional[str] = None,
    file_name: Optional[str] = "age_pyramid_uniqueness",
    **kwargs,
):
    df = pd.read_pickle(
        conf["tables_supp_material"]["output_paths"]["uniqueness_age_gender"]
    )
    population_pyramid(df, conf_name=conf_name, file_name=file_name)
    return Obj


@registry.plots("hospital_uniqueness")
def hospital_uniqueness(
    conf: Dict[str, Any],
    conf_name: Optional[str] = None,
    file_name: Optional[str] = "hospital_uniqueness",
    **kwargs,
):
    df = pd.read_pickle(
        conf["tables_supp_material"]["output_paths"]["uniqueness_hospital"]
    )
    bar_scatter_plot(
        df,
        y_bar="count",
        y_scatter="uniqueness",
        y_bar_label="Number of patients",
        y_scatter_label="Uniqueness",
        x_label="Hospital of the first hospitalisation",
        conf_name=conf_name,
        file_name=file_name,
    )
    return Obj


@registry.plots("stay_length_uniqueness")
def stay_length_uniqueness(
    conf: Dict[str, Any],
    conf_name: Optional[str] = None,
    file_name: Optional[str] = "stay_length_uniqueness",
    **kwargs,
):
    df = pd.read_pickle(
        conf["tables_supp_material"]["output_paths"]["uniqueness_stay_length"]
    )
    bar_scatter_plot(
        df,
        y_bar="count",
        y_scatter="uniqueness",
        y_bar_label="Number of patients",
        y_scatter_label="Uniqueness",
        x_label="Length of the first hospitalisation (days)",
        conf_name=conf_name,
        file_name=file_name,
    )
    return Obj


@registry.plots("n_stays_patients")
def n_stays_patients(
    conf: Dict[str, Any],
    conf_name: Optional[str] = None,
    file_name: Optional[str] = "n_stays_patients",
    **kwargs,
):
    df = pd.read_pickle(conf["tables_supp_material"]["output_paths"]["frequency_stays"])
    # cast to str type
    df.index = df.index.astype(str)

    bar_scatter_plot(
        df[:10],
        y_bar="count",
        y_scatter="uniqueness",
        y_bar_label="Number of patients",
        y_scatter_label="Uniqueness",
        x_label="Number of recorded stays",
        y_log_scale=True,
        conf_name=conf_name,
        file_name=file_name,
    )
    return Obj


def figureAge(cohort, stays, conf_general, variations_conf_table_1, indicator):
    if indicator.filter_first_visit:
        merge_cols = ["person_id"]
    else:
        merge_cols = ["person_id", "visit_occurrence_id"]

    for variation in variations_conf_table_1:
        conf_general.update(variation.copy())
        dataset = get_pseudonimized_dataset(cohort, stays, **conf_general)

        if variation["pseudonymization_algorithm"] == "NoPseudonymizer":
            indicator_output = indicator.compute(dataset)

            df = dataset.loc[indicator_output.index, merge_cols].copy()
            col_name = (
                "age_"
                + variation["pseudonymization_algorithm"]
                + "_"
                + str(variation["high_general"])
            )
            df[col_name] = indicator_output

        else:
            indicator_output = indicator.compute(dataset, shifted=True)

            df_tmp = dataset.loc[indicator_output.index, merge_cols].copy()
            col_name = (
                "age_"
                + variation["pseudonymization_algorithm"]
                + str(variation["high_general"])
            )
            df_tmp[col_name] = indicator_output

            df = df.merge(
                df_tmp,
                on=merge_cols,
                validate="one_to_one",
                how="left",
            )

    # Construct a df (one line per stay)
    df.set_index(merge_cols, inplace=True)
    schemas = [v["pseudonymization_algorithm"] for v in variations_conf_table_1]
    delta = [v["high_general"] for v in variations_conf_table_1]

    tuples = list(zip(*[schemas, delta]))

    df.columns = pd.MultiIndex.from_tuples(
        tuples, names=["pseudonymization_algorithm", "high_general"]
    )
    return df
