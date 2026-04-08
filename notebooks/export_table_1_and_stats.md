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
    display_name: privacy_local
    language: python
    name: privacy_local
---

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
```

```python
def read_table1(config, scenario, values=["reliability_indicator", "uniqueness"]):
    df = pd.read_csv(f"../data/{config}/{scenario}.csv")
    # df = pd.read_csv("../data/config_base/table1_random_target.csv")

    # df["pseudonymization_algorithm_parameter"] = df.pseudonymization_algorithm.str.cat(
    #     df.high_general.astype(str), sep=" - "
    # )

    df = df.query("cohort_name!='random'").copy()

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

    print("Cohort names", df.cohort_name.unique())

    print("N cohort max", df.n_cohort.max())

    print("N cohort min", df.n_cohort.min())

    print("## Mapping ##")
    print(cohort_name_mapping)

    df.cohort_name.replace(cohort_name_mapping, inplace=True)

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

    table = df.pivot(
        index=["pseudonymization_algorithm", "high_general"],
        columns="cohort_name",
        values=values,  # "success_rate" , "uniqueness"
    )

    table = table.swaplevel(
        axis=1,
    )

    table.sort_index(axis=1, inplace=True)

    table.sort_index(axis=0, inplace=True)

    table.sort_index(inplace=True)

    return table, df
```

```python
config = "config_seasonal_epidemics"
scenario = "table1_target_in_cohort"

table_se, df_se = read_table1(config=config, scenario=scenario)
```

```python
df_se = df_se.query("high_general.isin([7,30]) & cohort_name!='Overall'")
```

```python
table_se = table_se.loc[
    [
        ("BasePseudonymizer", 7),
        ("BasePseudonymizer", 30),
        ("BirthPseudonymizer", 7),
        ("BirthPseudonymizer", 30),
        ("StayPseudonymizer", 7),
        ("StayPseudonymizer", 30),
    ],
    [
        ("Seasonal bronchiolitis", "reliability_indicator"),
        ("Seasonal bronchiolitis", "uniqueness"),
        ("Seasonal flu", "reliability_indicator"),
        ("Seasonal flu", "uniqueness"),
    ],
]
table_se
```

```python
config = "config_base"
scenario = "table1_random_target"

table_base, df_base = read_table1(config=config, scenario=scenario)
```

```python
df_base.query("pseudonymization_algorithm == 'NoPseudonymizer'")
```

```python
table_base
```

```python
df = pd.concat(
    [
        df_base,
    ]
)  # df_se
```

```python
table = df.pivot(
    index=["pseudonymization_algorithm", "high_general"],
    columns="cohort_name",
    values=["reliability_indicator", "uniqueness"],  # "success_rate" , "uniqueness"
)

table = table.swaplevel(
    axis=1,
)

table.sort_index(axis=1, inplace=True)

table.sort_index(axis=0, inplace=True)

table.sort_index(inplace=True)
```

```python
table
```

```python
print(table.to_csv())
```

# Stats variables

```python
df.success_rate.describe()
```

```python
df.query("cohort_name=='Overall'").success_rate.describe()
```

```python
df.query("cohort_name!='Overall'").success_rate.describe()
```

```python
df.success_rate.describe()
```

```python
df.uniqueness.describe()
```

```python
df.reliability_indicator.describe()
```

```python
df.query(
    " pseudonymization_algorithm != 'NoPseudonymizer' & cohort_name !=  'Overall'"
).success_rate.describe()
```

```python
df.query(
    " pseudonymization_algorithm != 'NoPseudonymizer' & cohort_name !=  'Overall'"
).success_rate.max()
```

```python
break
```

# Stats stays

```python
from confit import Config
```

```python
from privacy.misc.utils import DataModule
```

```python
conf = Config.from_disk("../configs/config_base.cfg", resolve=True)
```

```python
data = DataModule(conf)
```

```python
stays = data.all_population_stays
```

```python
len(stays)
```

```python
cohort = data.all_population_cohort
len(cohort)
```

```python
stays = stays.merge(cohort, on="person_id", how="inner")
```

```python
len(stays)
```

```python
from privacy.indicators.age import Age
```

```python
ages = Age().compute(stays.query("visit_number==1"))
```

```python
ages.mean()
```

```python
ages.median()
```

```python
ages.quantile(0.25)
```

```python
ages.quantile(0.75)
```

```python
data.all_population_stays.visit_start_date.max()
```

```python
data.all_population_stays.visit_start_date.min()
```

# Population cohort

```python
data.all_population_cohort.gender_source_value.value_counts(normalize=True)
```

```python
data.all_population_cohort.gender_source_value.value_counts(normalize=False)
```

```python
data.all_population_cohort.death_date.notna().value_counts(normalize=True)
```

```python
stays_per_patient = data.all_population_stays.groupby("person_id", as_index=True).size()
```

```python
stays_per_patient.mean()
```

```python
stays_per_patient.median()
```

```python
stays_per_patient.quantile(0.25)
```

```python
stays_per_patient.quantile(0.75)
```

```python
stays_per_patient.value_counts(normalize=False)
```
