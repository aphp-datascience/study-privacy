[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Quantifying the effects of pseudonymisation on epidemiological research reliability: a tailored evaluation using a clinical data warehouse

Article available at : https://doi.org/10.1186/s12911-026-03360-0

## Abstract

### Background
Electronic health records (EHRs) hold immense potential for advancing medical research, but protecting patient privacy remains a critical challenge. Consequently, the choice of privacy-enhancing techniques must take into account the downstream analyses to preserve relevant data properties, often resulting in a trade-off between data utility and privacy. We aimed to evaluate different pseudonymisation algorithms and their impact in the context of six representative archetypal electronic health record epidemiological studies. This work seeks to empower Clinical Data Warehouse (CDW) stakeholders to make informed decisions that minimise privacy risks while ensuring information utility.

### Methods
We simulated various re-identification attempts conducted by an attacker with legitimate access to cohorts contained in the CDW of the Greater Paris University Hospitals. The dataset comprised 3,950,145 hospitalisation records with an admission between August 1st, 2017 and April 1st, 2024. We considered minimisation and pseudonymisation schemes with different parameterisations, randomly shifting the timestamps of the delivered data while preserving different degrees of temporal coherence among them. The impact of these techniques was assessed both on reliability of six representative archetypal epidemiological studies and on records uniqueness. Two attack scenarios were considered: a random-target attack and a target-in-cohort attack. Advantages and limitations of the different schemes were compared according to the specific requirements of the considered studies.

### Results
Attack success rates varied widely – ranging from a median of 0.9% [IQR: 0.3%-9.4%] in the random-target scenario to 99% [IQR: 86%-100%] in the target-in-cohort scenario – with minimisation accounting for most of this variability. Although less effective, pseudonymisation provided an additional reduction in re-identification risk. However, achieving low uniqueness required substantial modifications to temporal coherence, compromising the reliability of certain epidemiological statistics.

### Conclusions
Pseudonymisation must therefore be combined with other solutions, in particular data minimisation, to provide optimal privacy protection within CDWs. Our findings highlight the need for tailored data protection strategies that align with specific study objectives to preserve data utility for epidemiological research. Our findings will help Institutional Review Boards and CDW governance bodies and teams in making informed decisions to mitigate privacy risks while maintaining information utility.


## Citing this project
Please cite the following paper (add link) when using this project: 

`Cohen, Ariel, Yannick Jacob, Gilles Chatellier, et al. « Quantifying the Effects of Pseudonymisation on Epidemiological Research Reliability: A Tailored Evaluation Using a Clinical Data Warehouse ». BMC Medical Informatics and Decision Making 26, nᵒ 1 (2026): 87. https://doi.org/10.1186/s12911-026-03360-0.`

```
@article{cohen_quantifying_2026,
	title = {Quantifying the effects of pseudonymisation on epidemiological research reliability: a tailored evaluation using a clinical data warehouse},
	volume = {26},
	issn = {1472-6947},
	shorttitle = {Quantifying the effects of pseudonymisation on epidemiological research reliability},
	url = {https://link.springer.com/10.1186/s12911-026-03360-0},
	doi = {10.1186/s12911-026-03360-0},
	language = {en},
	number = {1},
	urldate = {2026-04-08},
	journal = {BMC Medical Informatics and Decision Making},
	author = {Cohen, Ariel and Jacob, Yannick and Chatellier, Gilles and Jean, Charline and Playe, Benoît and Mouchet, Alexandre and Audureau, Etienne and Boutet, Antoine and Bey, Romain},
	month = feb,
	year = {2026},
	pages = {87},
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

#### To make a slurm kernel with more RAM
```bash
slurm-kernel launch t4
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
python scripts/table1.py --config configs/config_base.cfg --scenario "target_in_cohort" --output-path "/export/home/acohen/privacy/data/config_base/table1_target_in_cohort.csv"
python scripts/table1.py --config configs/config_seasonal_epidemics.cfg
python scripts/table1.py --config configs/config_seasonal_epidemics.cfg --scenario "target_in_cohort" --output-path "/export/home/acohen/privacy/data/config_seasonal_epidemics/table1_target_in_cohort.csv"
sbatch scripts/sbatch_table_knowledge_uniqueness.sh
sbatch scripts/sbatch_table_simultaneous_variations_uniqueness.sh
python scripts/tables_supp_material.py --config configs/config_base.cfg
python scripts/make_plots.py --config configs/config_base.cfg
python scripts/make_plots2.py --config configs/config_base.cfg
python scripts/make_cluster_description.py --config configs/config_base.cfg
```