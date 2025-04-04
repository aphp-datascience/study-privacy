from typing import Optional

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from matplotlib import patches as mpatches
from matplotlib.font_manager import FontProperties

from privacy.plots.utils import show_or_save
from privacy.misc.constants import pseudonimizer_name_mapping


def add_handles_to_ax_legend_km(
    ax,
    position=1,
    label_title=r"Shift Parameter (days)",
    label_dashed_line=r"50% survival",
    font_properties=FontProperties(weight="bold", size=10),
):
    handles, _ = ax.get_legend_handles_labels()

    # Title patch
    title_shift_dates = mpatches.Patch(
        color=None,
        label=label_title,
        fill=False,
    )

    handles.insert(position, title_shift_dates)

    # dashed line
    line_handle = mlines.Line2D(
        [],
        [],
        color="black",
        # marker="-",
        linestyle="dashed",
        # markersize=10,
        label=label_dashed_line,
    )

    handles.insert(0, line_handle)

    ax.legend_ = ax.legend(handles=handles)

    # Font
    for text in ax.legend_.get_texts():
        if text.get_text() in [
            label_title,
        ]:
            text.set_fontproperties(font_properties)

    return ax


def km_plot(
    variations,
    conf_name: Optional[str] = None,
    file_name: Optional[str] = None,
    **kwargs,
):
    # Figure params
    figsize = (20, 5)
    fig, (ax1, ax2, ax3) = plt.subplots(
        1,
        3,
        figsize=figsize,
    )

    axes = {
        "BasePseudonymizer": ax1,
        "BirthPseudonymizer": ax2,
        "StayPseudonymizer": ax3,
    }

    loc = slice(0.0, 1500.0)
    for variation in variations:
        pseudonymization_algorithm = variation.get("pseudonymization_algorithm")
        data_survival = variation.get("indicator_output")
        label = (
            "No pseudonymisation"
            if pseudonymization_algorithm == "NoPseudonymizer"
            else str(variation.get("high_general"))
        )

        # Fit Kaplan Meier
        kmf = KaplanMeierFitter()

        kmf.fit(
            data_survival["observed_time"],
            event_observed=data_survival["death"],
            label=label,
        )
        if pseudonymization_algorithm == "NoPseudonymizer":
            for ax in axes.values():
                kmf.plot_survival_function(
                    ax=ax,
                    ci_show=False,
                    loc=loc,
                )
        else:
            ax = axes.get(pseudonymization_algorithm)
            kmf.plot_survival_function(
                ax=ax,
                ci_show=False,
                loc=loc,
            )

            ax.title.set_text(
                pseudonimizer_name_mapping.get(pseudonymization_algorithm)
            )
            ax.set_xlabel("No. of days since diagnosis")

        # add_at_risk_counts(
        #     kmf, ax=ax, rows_to_show=["At risk"], xticks=[0, 600, 1200], ypos=0
        # )

    for ax in fig.axes:
        ax.hlines(
            y=0.5, xmin=loc.start, xmax=loc.stop, colors="black", linestyles="dashed"
        )
        # ax.set_ylim(bottom=0)

        lines = ax.lines

        lines[0].set_linestyle((0, (6, 2)))
        lines[0].set_linewidth(2)
        lines[1].set_linestyle((1, (2, 4)))
        lines[1].set_linewidth(1.8)
        lines[2].set_linestyle((0, (2, 6)))
        lines[2].set_linewidth(1.7)
        ax = add_handles_to_ax_legend_km(ax)

    fig = fig.figure

    # Show or save plot
    show_or_save(
        fig,
        filename=file_name,
        conf_name=conf_name,
        # legend=[
        #     legend,
        # ],
    )
    return fig
