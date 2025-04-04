from typing import Optional

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from privacy.plots.utils import markers, show_or_save


class ProportionsPlot:
    def __init__(
        self,
        x="high_general",
        y="indicator_output",
        ylabel="3 months re-hospitalisation (%)",
        xlabel="Δt (days)",
        **kwargs,
    ) -> None:
        self.x = x
        self.y = y
        self.ylabel = ylabel
        self.xlabel = xlabel

    def preprocess_for_proportion_plot(self, df):
        idx = df.query("pseudonymization_algorithm=='NoPseudonymizer'").index
        p0 = df.loc[idx].indicator_output.iloc[0] * 100
        df = df.drop(index=idx, inplace=False)
        df.high_general = pd.Categorical(df.high_general)
        df[self.y] = df[self.y] * 100
        return df, p0

    def plot_proportions(
        self,
        df,
        p0,
        h=8,
        x="high_general",
        y="indicator_output",
        ylabel="Percentage of 3 months re-hospitalisation (%)",
        xlabel="Δt (days)",
    ):
        """
        df, p0 = preprocess_for_proportion_plot(df)
        plot_proportions(df, p0)
        """
        figsize = (h, h)
        fig, ax = plt.subplots(
            figsize=figsize,
        )

        fig = sns.barplot(
            df,
            x=x,
            y=y,
            hue="pseudonymization_algorithm",
            # style="pseudonymization_algorithm",
            # markers=markers,
            ax=ax,
            # s=130,
        )
        # ax.set(
        #     xscale="log",
        # )
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.hlines(
            y=p0,
            # xmin=0,
            # xmax=df[x].cat.as_ordered().max(),
            xmin=-0.75,
            xmax=2.75,
            colors="black",
            linestyles="dashed",
        )
        ax.set_ylim(bottom=0)
        # ax.text(
        #     6.4,
        #     p0 - 0.002,
        #     "p0",
        # )

        handles, _ = ax.get_legend_handles_labels()
        title_shift_dates = mlines.Line2D(
            [],
            [],
            color="black",
            linestyle="dashed",
            markersize=15,
            label="No pseudonymization",
        )
        handles.insert(0, title_shift_dates)

        ax.legend_ = ax.legend(handles=handles)
        ax.legend_.set_title("Pseudonymization algorithm")

        fig = fig.figure
        return fig

    def __call__(
        self,
        results,
        conf_name: Optional[str] = None,
        file_name: Optional[str] = None,
        **kwargs,
    ):
        df, p0 = self.preprocess_for_proportion_plot(pd.DataFrame(results))
        fig = self.plot_proportions(
            df=df, p0=p0, x=self.x, y=self.y, ylabel=self.ylabel, xlabel=self.xlabel
        )

        # Show or save plot
        show_or_save(
            fig,
            filename=file_name,
            conf_name=conf_name,
            # legend=[
            #     legend,
            # ],
        )
