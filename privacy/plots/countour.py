import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas import DataFrame

from privacy.plots.utils import *

sns.set_theme(style="whitegrid")


def countour_subplot(
    x,
    y,
    z,
    ax,
    levels=10,
    xlabel="Δt (days)",
    ylabel="Δt birth (days)",
    zlabel="Uniqueness",
    log_scale: bool = False,
):
    CS = ax.contour(x, y, z, levels=levels)
    ax.clabel(CS, inline=True, fontsize=10)
    if log_scale:
        ax.set(yscale="log", xscale="log")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # make a colorbar for the contour lines
    CB = plt.colorbar(
        mappable=CS,
        ax=ax,
        shrink=0.8,
        label=zlabel,
    )


def get_x_y_z(df_pivot):
    x = df_pivot.index.values  # clinical shift

    y = df_pivot.columns.values  # birth shift
    X, Y = np.meshgrid(x, y)
    Z = df_pivot.values

    return X, Y, Z


def uniqueness_dt_subplot(
    ax,
    df: DataFrame,
    x: str,
    hue: str,
    y: str = "uniqueness",
    h: int = 5,
    ylabel="Uniqueness",
):
    ax.set(
        xscale="log",
    )

    palette = sns.color_palette("rocket_r", as_cmap=True)
    sns.lineplot(
        data=df,
        ax=ax,
        x=x,
        y=y,
        hue=hue,
        palette=palette,
    )

    sns.move_legend(
        ax,
        "upper left",
        bbox_to_anchor=(1, 1),
        title=None,
        frameon=True,
    )
    ax.set_ylim(0)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(mapping_dt[x])
    ax.legend(title=mapping_dt[hue])
