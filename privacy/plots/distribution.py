from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

from privacy.plots.utils import add_handle_to_ax_legend_at_position, show_or_save
from privacy.misc.constants import pseudonimizer_name_mapping


class DistributionPlot:
    def __init__(self, **kwargs) -> None:
        pass

    def preprocess_for_distribution_plot(self, results):
        distributions = [o["indicator_output"] for o in results]
        schemas = [v["pseudonymization_algorithm"] for v in results]
        delta = [v["high_general"] for v in results]
        tuples = list(zip(*[schemas, delta]))

        df = pd.DataFrame(distributions).T

        df.columns = pd.MultiIndex.from_tuples(
            tuples, names=["pseudonymization_algorithm", "high_general"]
        )
        df2 = df.melt(value_vars=list(df.columns), ignore_index=False)
        df2["value"] = df2["value"].astype(int)
        df2.index.name = "week"
        df2.reset_index(inplace=True)
        return df2

    def rename_xticks(
        self,
        ax1,
        ticks_labels=[
            "",
            "January",
            "March",
            "May",
            "July",
            "September",
            "December",
            "",
        ],
        labelrotation=45,
    ):
        ticks_loc = ax1.get_xticks().tolist()

        ax1.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax1.set_xticklabels(ticks_labels)

        ax1.tick_params(axis="x", labelrotation=labelrotation)

    def plot_distribution(
        self,
        df,
        h=8,
        x="week",
        y="value",
        ylabel="No. Hospitalisations",
        xlabel="",
    ):
        """
        df, p0 = preprocess_for_proportion_plot(df)
        plot_proportions(df, p0)
        """
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

        idx = df.pseudonymization_algorithm == "NoPseudonymizer"
        df_no_pseudo = df.loc[idx]
        df_pseudo = df.loc[~idx]

        for pseudonymization_algorithm, ax in axes.items():
            ax = sns.lineplot(
                pd.concat(
                    [
                        df_no_pseudo,
                        df_pseudo.query(
                            "pseudonymization_algorithm==@pseudonymization_algorithm"
                        ),
                    ]
                ),
                x=x,
                y=y,
                hue="high_general",
                palette=sns.color_palette(n_colors=4, as_cmap=False),
                ax=ax,
            )
            ax.title.set_text(
                pseudonimizer_name_mapping.get(pseudonymization_algorithm)
            )
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            ax = add_handle_to_ax_legend_at_position(ax)
            for text in ax.legend_.get_texts():
                if text.get_text() == "0":
                    text.set_text("No pseudonymisation")

            self.rename_xticks(ax)

        fig = fig.figure

        return fig

    def __call__(
        self,
        results,
        conf_name: Optional[str] = None,
        file_name: Optional[str] = None,
        **kwargs,
    ):
        df = self.preprocess_for_distribution_plot(results)
        fig = self.plot_distribution(df)
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
