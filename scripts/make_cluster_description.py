import pandas as pd
from privacy.misc.utils import DataModule
from confit import Config
from privacy.pipelines.reliability import get_pseudonimized_dataset
from privacy.indicators.age import Age
from privacy.indicators.cluster import Cluster
from privacy.pipelines.table1 import variations_conf_table_1
from privacy.pipelines.plots import figureAge
from privacy.misc.utils import build_path

from confit import Cli

from privacy import registry
from typing import Any, Dict, List
app = Cli(pretty_exceptions_show_locals=False)



@app.command("") #"cancer"
def main(
    
    config_meta=None,
):
    conf_name = config_meta["config_path"][0].stem
    conf = config_meta["resolved_config"]

    cohort_name ="cancer"
    dm = DataModule(conf)
    stays = dm.stays(cohort_name)
    cohort = dm.cohort(cohort_name)
    print(f"################################ {cohort_name} #######################")
    print(cohort_name, "size", len(cohort))
    
    seed = conf["cancer"]["clustering"].seed
    print("seed",seed)

   
    indicators = conf[cohort_name]["indicators"]

    print("indicators", indicators)

    for indicator in indicators:
        print("Indicator Class:",indicator.__class__)

        # Set conf dictionary
        conf_general = {"seed": seed, "cohort_name":cohort_name}      
        if indicator.__class__ == Cluster:
            results = []
            for variation in variations_conf_table_1:
                # Set conf dictionary
                conf_general = {"seed": seed, "cohort_name":cohort_name}
                conf_general.update(variation.copy())

                # Pseudonymize
                dataset = get_pseudonimized_dataset(cohort, stays, **conf_general)
                if variation["pseudonymization_algorithm"] == "NoPseudonymizer":
                    (
                        cluster_description,
                        sample,
                        patients_largest_cluster_non_pseudo,
                        indicator_output,
                    ) = indicator.compute(dataset)
                else:
                    (
                        cluster_description,
                        _,
                        _,
                        indicator_output,
                    ) = indicator.compute(
                        dataset,
                        patients_largest_cluster_non_pseudo=patients_largest_cluster_non_pseudo,
                    )

                conf_general.update(
                    dict(
                        cluster_description=cluster_description,
                        cohort_name=cohort_name,
                        indicator_output=indicator_output,
                    )
                )
                results.append(conf_general.copy())
                
                print("indicator:",indicator_output)
            results = pd.DataFrame(results)

            for _, row in results.iterrows():
                row.cluster_description["pseudonymization_algorithm"] = row[
                    "pseudonymization_algorithm"
                ]
                row.cluster_description["high_general"] = row["high_general"]
            
            cluster_description = pd.concat(results.cluster_description.to_list())
            
            

            # results.drop(columns="cluster_description").to_csv(build_path(__file__,f"../data/{conf_name}/{cohort_name}/cluster_indicator.csv")) 
            cluster_description.to_csv(build_path(__file__,f"../data/{conf_name}/{cohort_name}/cluster_description.csv"))
            print("saved!")
            
        
if __name__ == "__main__":
    app()