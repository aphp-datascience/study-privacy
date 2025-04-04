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
df = pd.read_csv("../data/config_base/table1.csv")
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
    columns="cohort_name",
    values=["reliability_indicator", "success_rate"],
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

# Stats variables

```python
df.success_rate.quantile(q=0.25) * 100
```

```python
df.success_rate.median() * 100
```

```python
df.success_rate.quantile(q=0.75) * 100
```

```python
df.query("cohort_name=='Overall'").success_rate.quantile(q=0.75)
```

```python
df.query("cohort_name=='Overall'").success_rate.median()
```

```python
df.query("cohort_name=='Overall'").success_rate.quantile(q=0.25)
```

```python
df.query("cohort_name!='Overall'").success_rate.median()
```

```python
df.query("cohort_name!='Overall'").success_rate.quantile(q=0.75)
```

```python
df.query("cohort_name!='Overall'").success_rate.quantile(q=0.25)
```

```python
df.success_rate.min()
```

```python
df.success_rate.max()
```

```python
df.uniqueness.median()
```

```python
df.uniqueness.quantile(q=0.75)
```

```python
df.uniqueness.min()
```

```python
df.uniqueness.max()
```

```python
df.uniqueness.quantile(q=0.25)
```

```python
df.reliability_indicator.max()
```

```python
df.query(
    " pseudonymization_algorithm != 'NoPseudonymizer' & cohort_name !=  'Overall'"
).success_rate.min()
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
