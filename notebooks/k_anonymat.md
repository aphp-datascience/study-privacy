---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.0
  kernelspec:
    display_name: privacy_python
    language: python
    name: privacy_python
---

```python
%config Completer.use_jedi = False
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
from privacy import registry

from confit import Config

import pandas as pd

from privacy.misc.utils import DataModule

from privacy.attacks.uniqueness import Uniqueness

from privacy.pipelines.uniqueness import pipe_uniqueness

import polars as pl

from privacy.pipelines.table1 import get_table_1
import ast
```

```python
import sys
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")
```

```python
conf = Config.from_disk("../configs/config_base.cfg", resolve=True)
```

```python
data = DataModule(conf=conf)
```

```python
cohorts_definitions = {
    "all_population": conf.get("cohorts")
    .get("cohorts_definitions")
    .get("all_population"),
    "pancreatic_cancer": conf.get("cohorts")
    .get("cohorts_definitions")
    .get("pancreatic_cancer"),
}
cohorts_definitions.keys()
```

```python
attack_knowledge = conf.get("table1").get("attack_knowledge")
```

```python
path_table = "/export/home/acohen/privacy/data/config_base/table_k_anonymity.csv"
```

```python
# path_table = "/export/home/acohen/privacy/data/config_base/table_k_anonymity.csv"
# table_k_anonymity = get_table_1(
#     cohorts_definitions=cohorts_definitions,
#     data=data,
#     attack_knowledge=attack_knowledge,
#     seed=conf["general"]["seed"],
#     output_path=path_table,
# )
# table_k_anonymity.to_csv(path_table)
```

```python
table_k_anonymity = pd.read_csv(path_table)
```

```python
table_k_anonymity.remainder = table_k_anonymity.remainder.apply(
    lambda x: ast.literal_eval(x)
)
```

```python
table_k_anonymity.remainder.iloc[1]
```

```python
import pandas as pd


def aggregate_k_anonymity(k_anonymity_dict, partition):
    """
    Aggregate k_anonymity counts into categories defined by partition.
    partition = [0, 1, 5, 10, 100, 1000, 100000] creates categories:
    [0-1), [1-5), [5-10), [10-100), [100-1000), [1000-100000)
    """
    if pd.isna(k_anonymity_dict) or not k_anonymity_dict:
        return {
            f"[{partition[i]}-{partition[i+1]})": 0 for i in range(len(partition) - 1)
        }

    # Initialize category counts

    category_counts = {
        f"[{partition[i]}-{partition[i+1]})": 0 for i in range(0, len(partition) - 1)
    }

    # Aggregate each k_anonymity into appropriate category
    for k_anonymity, count in k_anonymity_dict.items():
        for i in range(len(partition) - 1):
            if partition[i] <= k_anonymity < partition[i + 1]:
                category_counts[f"[{partition[i]}-{partition[i+1]})"] += count
                break

    return category_counts
```

```python
# Apply
partition = [1, 2, 5, 10, 100, 1000, 100000]
table_k_anonymity["k_anonymity_categories"] = table_k_anonymity["remainder"].apply(
    lambda x: aggregate_k_anonymity(x, partition)
)
```

```python
table_k_anonymity.iloc[1].k_anonymity_categories
```

```python
k_anonymity = pd.DataFrame(table_k_anonymity.k_anonymity_categories.to_list())
```

```python
k_anonymity = pd.concat(
    [
        table_k_anonymity[
            ["pseudonymization_algorithm", "high_general", "uniqueness", "cohort_name"]
        ],
        k_anonymity,
    ],
    axis=1,
)
```

```python
k_anonymity.rename(columns={"[1-2)": "1"}, inplace=True)
```

```python
count_columns = [
    "1",
    "[2-5)",
    "[5-10)",
    "[10-100)",
    "[100-1000)",
    "[1000-100000)",
]
```

```python
n_patients_to_check = conf.get("general").get("n_patients_to_check")
n_patients_to_check
```

```python
k_anonymity[count_columns] = k_anonymity[count_columns] / n_patients_to_check
```

```python
k_anonymity
```

```python
k_anonymity.columns
```

```python
p_dataset_5_plus = k_anonymity.query(
    "cohort_name=='all_population' & pseudonymization_algorithm!= 'NoPseudonymizer' "
)[["[5-10)", "[10-100)", "[100-1000)", "[1000-100000)"]].sum(axis=1)
p_dataset_5_plus
```

```python
1 - p_dataset_5_plus.describe()
```

```python
# print(k_anonymity.to_markdown())
```

```python
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

from privacy.plots.utils import add_handle_to_ax_legend_at_position, show_or_save
from privacy.misc.constants import pseudonimizer_name_mapping

sns.set_theme(style="whitegrid")


class DistributionPlot:
    def __init__(
        self,
        count_columns=[
            "1",
            "[2-5)",
            "[5-10)",
            "[10-100)",
            "[100-1000)",
            "[1000-100000)",
        ],
        **kwargs,
    ) -> None:
        self.count_columns = count_columns

    def preprocess_for_distribution_plot(self, df):
        # Melt to long format
        df2 = df.melt(
            id_vars=["pseudonymization_algorithm", "high_general"],
            value_vars=self.count_columns,
            var_name="age_category",
            value_name="value",
        )
        # df2["value"] = df2["value"].astype(int)

        return df2

    def rename_xticks(
        self,
        ax1,
        ticks_labels=None,
        labelrotation=45,
    ):
        if ticks_labels is None:
            # Keep original age category labels
            return

        ticks_loc = ax1.get_xticks().tolist()
        ax1.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax1.set_xticklabels(ticks_labels)
        ax1.tick_params(axis="x", labelrotation=labelrotation)

    def plot_distribution(
        self,
        df,
        h=8,
        x="age_category",
        y="value",
        ylabel="Relative frequency",
        xlabel="k-anonymity",
    ):
        """
        Plot age category distributions as bar plots
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
            combined_df = pd.concat(
                [
                    df_no_pseudo,
                    df_pseudo.query(
                        "pseudonymization_algorithm==@pseudonymization_algorithm"
                    ),
                ]
            )

            ax = sns.barplot(
                combined_df,
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

            ax.tick_params(axis="x", labelrotation=45)

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
        )
        return fig
```

```python
for cohort_name in k_anonymity.cohort_name.unique():
    print(f"#### {cohort_name} ####")
    _ = DistributionPlot()(
        results=k_anonymity.loc[k_anonymity.cohort_name == cohort_name],
        conf_name="config_base",
        file_name=f"k_anonymity_{cohort_name}",
    )
```

```python
# _ = DistributionPlot()(k_anonymity.query("cohort_name=='pancreatic_cancer'"))
```

```python

```
