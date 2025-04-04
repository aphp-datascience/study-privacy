set -e
export ARROW_LIBHDFS_DIR=/usr/local/hadoop/usr/lib/
export HADOOP_HOME=/usr/local/hadoop
export CLASSPATH=`$HADOOP_HOME/bin/hdfs classpath --glob`


# bash scripts/spark_submit.sh scripts/cohort_generator.py --config configs/config_base.cfg --cohorts.cohort="all_population"
# bash scripts/spark_submit.sh scripts/cohort_generator.py --config configs/config_base.cfg --cohorts.cohort="bronchiolitis"
# bash scripts/spark_submit.sh scripts/cohort_generator.py --config configs/config_base.cfg --cohorts.cohort="seasonal_flu"
# bash scripts/spark_submit.sh scripts/cohort_generator.py --config configs/config_base.cfg --cohorts.cohort="bariatric_surgery"
# bash scripts/spark_submit.sh scripts/cohort_generator.py --config configs/config_base.cfg --cohorts.cohort="cancer"
# bash scripts/spark_submit.sh scripts/cohort_generator.py --config configs/config_base.cfg --cohorts.cohort="pancreatic_cancer"

# python scripts/table1.py --config configs/config_base.cfg
# sbatch scripts/sbatch_table_knowledge_uniqueness.sh
sbatch scripts/sbatch_table_simultaneous_variations_uniqueness.sh
python scripts/tables_supp_material.py --config configs/config_base.cfg
python scripts/make_plots.py --config configs/config_base.cfg