from typing import Optional

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import seaborn as sns

from privacy.plots.utils import show_or_save
from privacy.misc.constants import pseudonimizer_name_mapping


class BoxPlot:
    def __init__(
        self,
        h=8,
        x="high_general",
        y="value",
        ylabel="Age at admission (years)",
        xlabel="Δt (days)",
        title=None,
        compute_difference=False,
        whis=None,
        showfliers=False,
        **kwargs,
    ) -> None:
        self.x = x
        self.h = h
        self.y = y
        self.ylabel = ylabel
        self.xlabel = xlabel
        self.title = title
        self.compute_difference = compute_difference
        self.whis = whis
        self.showfliers = showfliers

    def preprocess_for_box_plot(self, df):
        age0 = df.loc[:, ("NoPseudonymizer", 0)].copy()
        df2 = df.drop(inplace=False, columns=("NoPseudonymizer", 0))

        if self.compute_difference:
            df2 = df2.subtract(age0, axis=0)
        else:
            # df2[("NoPseudonymizer", 10)] = age0
            # df2[("NoPseudonymizer", 100)] = age0
            # df2[("NoPseudonymizer", 1000)] = age0
            df2[("NoPseudonymizer", 0)] = age0

        df3 = df2.melt(value_vars=list(df2.columns))

        df3["pseudonymization_algorithm"].replace(
            pseudonimizer_name_mapping, inplace=True
        )

        return df3

    def plot_box(
        self,
        df,
    ):
        """
        df, p0 = preprocess_for_proportion_plot(df)
        plot_proportions(df, p0)
        """

        x = self.x
        h = self.h
        y = self.y
        ylabel = self.ylabel
        xlabel = self.xlabel
        title = self.title
        whis = self.whis
        showfliers = self.showfliers

        figsize = (h, h)
        fig, ax = plt.subplots(
            figsize=figsize,
        )

        fig = sns.boxplot(
            df,
            x=x,
            y=y,
            hue="pseudonymization_algorithm",
            ax=ax,
            hue_order=list(
                pseudonimizer_name_mapping.values()
            ),  # ['NoPseudonymizer', 'BasePseudonymizer', 'BirthPseudonymizer','StayPseudonymizer'],
            whis=whis,
            showfliers=showfliers,
        )

        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

        if self.compute_difference:
            ax.hlines(
                y=0,
                xmin=-0.75,
                xmax=2.75,
                colors="black",
                linestyles="dashed",
            )

            handles, _ = ax.get_legend_handles_labels()
            title_shift_dates = mlines.Line2D(
                [],
                [],
                color="red",
                # marker="-",
                linestyle="dashed",
                markersize=15,
                label="No pseudonymization",
            )
            handles.insert(0, title_shift_dates)

            ax.legend_ = ax.legend(handles=handles)
            ax.legend_ = ax.legend(loc="upper right")

        ax.legend_.set_title("Pseudonymization algorithm")
        ax.legend(bbox_to_anchor=(0.8, -0.1), ncol=2)
        if title is not None:
            ax.set_title(title)

        fig = fig.figure
        return fig

    def __call__(
        self,
        results,
        conf_name: Optional[str] = None,
        file_name: Optional[str] = None,
        **kwargs,
    ):
        df = self.preprocess_for_box_plot(results)
        fig = self.plot_box(df)

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
