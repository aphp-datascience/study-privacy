import os
from typing import List, Optional

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches as mpatches
from matplotlib.font_manager import FontProperties
from pandas import DataFrame

from privacy.misc.utils import build_path

palette = sns.color_palette(["#C4C4C4", "#646464", "#000000"], as_cmap=True)


def legend_rename(key):
    if key == "high_general":
        return "Shift Parameter (days)"

    elif key == "pseudonymization_algorithm":
        return "Pseudonymization Algorithm"
    else:
        return key


markers = {
    "NoPseudonymizer": "D",
    "No pseudonymisation": "D",
    "BasePseudonymizer": "v",
    "Base pseudonymisation": "v",
    "BirthPseudonymizer": "^",
    "Birth pseudonymisation": "^",
    "StayPseudonymizer": "X",
    "Hospital stay pseudonymisation": "X",
}


def _save_plot(
    fig,
    filename: str,
    conf_name: str,
    legend=None,
    tables: Optional[List[DataFrame]] = None,
    formats=["pdf", "png"],
    project_name: str = "privacy",
    save_index: bool = False,
    **kwargs,
):
    """
    Auxiliary function to save plot.
    """
    folder = os.path.expanduser(f"~/{project_name}/figures/{conf_name}")
    filenameimgs = [filename + "_" + conf_name + "." + format for format in formats]

    path_dir = build_path(__file__, folder)
    path_files = [os.path.join(path_dir, filenameimg) for filenameimg in filenameimgs]

    path_dir_extended = os.path.dirname(path_files[0])
    if not os.path.isdir(path_dir_extended):
        os.makedirs(path_dir_extended)

    for path_file, format in zip(path_files, formats):
        if legend is not None:
            fig.savefig(
                path_file,
                bbox_extra_artists=tuple(legend),
                bbox_inches="tight",
                format=format,
            )
        else:
            fig.savefig(
                path_file,
                format=format,
                bbox_inches="tight",
            )

        print("Saved at:", path_file)

    plt.cla()
    plt.close("all")

    if tables:
        for i, table in enumerate(tables, start=1):
            filenametable = filename + "_" + str(i) + "_" + conf_name + ".csv"
            path_table = os.path.join(path_dir, filenametable)
            table.to_csv(path_table, index=save_index)

    print("Done -", filename)


def show_or_save(
    fig,
    filename: Optional[str] = None,
    conf_name: Optional[str] = None,
    legend=None,
    tables: Optional[List[DataFrame]] = None,
    **kwargs,
):
    # Show or save plot
    if (conf_name is not None) & (filename is not None):
        _save_plot(
            fig,
            filename=filename,
            conf_name=conf_name,
            legend=legend,
            tables=tables,
            **kwargs,
        )

    else:
        plt.show()
        plt.cla()
        plt.close("all")


mapping_dt = {
    "high_birth_date": "Δt birth (days)",
    "high_general": "Δt (days)",
}


def add_handle_to_ax_legend_at_position(
    ax,
    position=1,
    label=r"Shift Parameter (days)",
    font_properties=FontProperties(weight="bold", size=10),
):
    handles, _ = ax.get_legend_handles_labels()
    title_shift_dates = mpatches.Patch(
        color=None,
        label=label,
        fill=False,
    )

    handles.insert(position, title_shift_dates)

    ax.legend_ = ax.legend(handles=handles)

    for text in ax.legend_.get_texts():
        if text.get_text() in [
            label,
        ]:
            text.set_fontproperties(font_properties)

    return ax


def add_line_legend_at_end(ax, label):
    line_handle = mlines.Line2D(
        [],
        [],
        color="black",
        # marker="-",
        linestyle="dashed",
        # markersize=10,
        label=label,
    )
    handles, _ = ax.get_legend_handles_labels()

    handles.append(line_handle)

    ax.legend_ = ax.legend(handles=handles)

    return ax
