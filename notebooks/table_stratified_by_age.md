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

```

```python
import pandas as pd
```

```python
%load_ext autoreload
%autoreload 2
%config Completer.use_jedi = False
%load_ext jupyter_black
import pandas as pd

pd.set_option("max_columns", None)
```

```python
from privacy.misc.constants import cohort_name_mapping
from pathlib import Path
```

```python
paths = [
    "table_1_by_age_cat_60-74.csv",
    "table_1_by_age_cat_90+.csv",
    "table_1_by_age_cat_0-14.csv",
    "table_1_by_age_cat_45-59.csv",
    "table_1_by_age_cat_75-89.csv",
    "table_1_by_age_cat_15-29.csv",
    "table_1_by_age_cat_30-44.csv",
]
```

```python
df_list = []
for path in paths:
    tmp = pd.read_csv(Path("../data/config_base/", path))

    df_list.append(tmp)
df = pd.concat(df_list, axis=0)
```

```python
df
```

```python
ri = pd.read_csv(
    "../data/config_base/results_reliability_indicator_by_age_corrected.csv"
)
```

```python
ri = ri[
    [
        "pseudonymization_algorithm",
        "high_general",
        "age_category",
        "reliability_indicator",
    ]
]
```

```python
df = df.rename(
    columns={
        "reliability_indicator": "old_reliability_indicator",
    }
)
ri = ri.rename(
    columns={
        "age_category": "age_cat",
    }
)
```

```python
df = df.merge(
    ri,
    on=[
        "pseudonymization_algorithm",
        "high_general",
        "age_cat",
    ],
    how="left",
    validate="one_to_one",
)
```

```python
df
```

```python
# df["pseudonymization_algorithm_parameter"] = df.pseudonymization_algorithm.str.cat(
#     df.high_general.astype(str), sep=" - "
# )
```

```python
df = df.query("cohort_name!='random'").copy()
```

```python
df.pseudonymization_algorithm = pd.Categorical(
    df.pseudonymization_algorithm,
    categories=[
        "NoPseudonymizer",
        "BasePseudonymizer",
        "BirthPseudonymizer",
        "StayPseudonymizer",
    ],
    ordered=True,
)
```

```python
df.age_cat.unique()
```

```python
df.age_cat = pd.Categorical(
    df.age_cat,
    categories=[
        "0-14",
        "15-29",
        "30-44",
        "45-59",
        "60-74",
        "75-89",
        "90+",
    ],
    ordered=True,
)
```

```python
df.cohort_name.unique()
```

```python
df.n_cohort.max()
```

```python
df.n_cohort.min()
```

```python
cohort_name_mapping
```

```python
df.cohort_name.replace(cohort_name_mapping, inplace=True)
```

```python
df.cohort_name = pd.Categorical(
    df.cohort_name,
    categories=[
        "Overall",
        "Seasonal bronchiolitis",
        "Seasonal flu",
        "Bariatric surgery readmission",
        "Pancreatic Cancer",
        "Cancer",
    ],
    ordered=True,
)
```

```python
table = df.pivot(
    index=["pseudonymization_algorithm", "high_general"],
    columns="age_cat",
    values=["reliability_indicator", "uniqueness"],  # "success_rate"
)
```

```python
table = table.swaplevel(
    axis=1,
)
```

```python
table.sort_index(axis=1, inplace=True)
```

```python
table.sort_index(axis=0, inplace=True)
```

```python
table.sort_index(inplace=True)
```

```python
table
```

```python
print(table.to_csv())
```

```python
table.columns.get_level_values(1)
```

```python
filtered_table = table.loc[:, table.columns.get_level_values(1) == "uniqueness"]
```

```python
filtered_table.mean()
```

```python
age_cat = "90+"
```

```python
filtered_table.mean(axis=1)
```

```python
(filtered_table.loc[:, (age_cat, "uniqueness")])
```

```python
(
    filtered_table.loc[:, (age_cat, "uniqueness")] / filtered_table.mean(axis=1)
).describe()
```

```python
(filtered_table.loc[:, (age_cat, "uniqueness")] / filtered_table.mean(axis=1)).mean()
```

```python

```
