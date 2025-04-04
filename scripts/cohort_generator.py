from confit import Cli

from privacy import registry
from privacy.misc.utils import arrowConnector

app = Cli(pretty_exceptions_show_locals=False)
from typing import Any, Dict

from loguru import logger


@app.command(name="cohorts")
def main(
    cohorts_definitions: Any,
    cohort: str,
    path_tmp_hdfs_cohort: str,
    path_tmp_hdfs_stays: str,
):
    cohort_selector = cohorts_definitions[cohort]["cohort_selector"]
    path_save_cohort = cohorts_definitions[cohort]["path_save_cohort"]
    path_save_stays = cohorts_definitions[cohort]["path_save_stays"]

    # Retrieve Stays & Cohort
    vo, cohort = cohort_selector()

    # Stays
    vo.write.mode("overwrite").parquet(path_tmp_hdfs_stays)

    # Cohort
    cohort.write.mode("overwrite").parquet(path_tmp_hdfs_cohort)

    # Read and write in pickle
    connector = arrowConnector()
    df = connector.get_pd_table(path_tmp_hdfs_stays)
    df.to_parquet(path_save_stays)
    logger.info(f"Stays saved at : {path_save_stays}")

    df = connector.get_pd_table(path_tmp_hdfs_cohort)
    df.to_parquet(path_save_cohort)
    logger.info(f"Cohort saved at : {path_save_cohort}")


if __name__ == "__main__":
    app()
