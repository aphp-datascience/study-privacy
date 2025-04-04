[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Pseudonymisation and Epidemiological Research Reliability: a Tailored Approach Using a Clinical Data Warehouse

## Project presentation - Abstract

#### Background
Electronic health records (EHRs) hold immense potential for advancing medical research, but protecting patient privacy remains a critical challenge. However, the choice of the privacy enhancing techniques must take into account the downstream analyses in order to preserve relevant data properties, often resulting in a trade-off between data utility and privacy. We aimed to evaluate different pseudonymisation algorithms and their impact in the context of six representative archetypal electronic health record epidemiological studies in order to enable Clinical Data Warehouse (CDW) actors to make better decisions to minimise privacy risks while ensuring the information utility. 

#### Methods
We simulated various re-identification attempts conducted by an attacker with legitimate access to cohorts contained in the CDW of the Greater Paris University Hospitals. The dataset contained 3,950,145 hospitalisation records with an admission between August 1st, 2017 and April 1st, 2024. We considered minimisation and pseudonymisation schemes with different parameterisations, randomly shifting the timestamps of the delivered data while preserving different degrees of temporal coherence among them. We assessed the impact of these techniques both on the reliability of six representative archetypal epidemiological studies and on the uniqueness of the records. The advantages and limitations of the different schemes are compared according to the considered studies.

#### Findings
Attack success rates varied widely [median 0·9%, IQR 0·3%-9·4%] and minimisation accounted for most of this variability. Although less effective, pseudonymisation provided an additional reduction in re-identification risk. However, to achieve low uniqueness, temporal coherence had to be strongly modified, affecting the reliability of some epidemiological statistics. 

#### Interpretation
Pseudonymisation must therefore be combined with other solutions, in particular data minimisation, to optimally protect privacy in the context of CDWs. Our findings highlight the need for tailored data protection strategies that align with specific study objectives in order to ensure data utility.


## Citing this project
Please cite the following paper (add link) when using this project:

```
@inproceedings{project_title,
  title={},
  author={},
  booktitle = {},
  pages     = {},
  year      = {}
}
```

# How to run the code
## Install Python env
#### Create an environment
```bash
python -m venv .venv
```

#### Activate it
```bash
source .venv/bin/activate
```

#### Install packages
```bash
pip install pypandoc==1.7.5
pip install pyspark==2.4.8
poetry install
pip uninstall pypandoc
```

## Install R env
```bash
conda create -n r_env
conda activate r_env
conda install -c conda-forge r-base==4.2.2
R
install.packages("TraMineR", repos="https://cran.irsn.fr/")
/// install.packages("tidyverse", repos="https://cran.irsn.fr/")
quit()
conda install -c conda-forge r-irkernel
R
IRkernel::installspec(name="r_env", displayname="r_env")
quit()
```

## Run
```bash
cd privacy
conda deactivate
source .venv/bin/activate 
export ARROW_LIBHDFS_DIR=/usr/local/hadoop/usr/lib/
export HADOOP_HOME=/usr/local/hadoop
export CLASSPATH=`$HADOOP_HOME/bin/hdfs classpath --glob`


bash scripts/spark_submit.sh scripts/cohort_generator.py --config configs/config_base.cfg --cohorts.cohort="all_population"
bash scripts/spark_submit.sh scripts/cohort_generator.py --config configs/config_base.cfg --cohorts.cohort="bronchiolitis"
bash scripts/spark_submit.sh scripts/cohort_generator.py --config configs/config_base.cfg --cohorts.cohort="seasonal_flu"
bash scripts/spark_submit.sh scripts/cohort_generator.py --config configs/config_base.cfg --cohorts.cohort="bariatric_surgery"
bash scripts/spark_submit.sh scripts/cohort_generator.py --config configs/config_base.cfg --cohorts.cohort="cancer"
bash scripts/spark_submit.sh scripts/cohort_generator.py --config configs/config_base.cfg --cohorts.cohort="pancreatic_cancer"


python scripts/table1.py --config configs/config_base.cfg
sbatch scripts/sbatch_table_knowledge_uniqueness.sh
sbatch scripts/sbatch_table_simultaneous_variations_uniqueness.sh
python scripts/tables_supp_material.py --config configs/config_base.cfg
python scripts/make_plots.py --config configs/config_base.cfg
python scripts/make_plots2.py --config configs/config_base.cfg
python scripts/make_cluster_description.py --config configs/config_base.cfg
```