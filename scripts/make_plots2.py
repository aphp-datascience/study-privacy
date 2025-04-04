from privacy.misc.utils import DataModule
from privacy.pipelines.reliability import get_pseudonimized_dataset
from privacy.indicators.age import Age
from privacy.indicators.cluster import Cluster
from privacy.pipelines.table1 import variations_conf_table_1
from privacy.pipelines.plots import figureAge
from typing import List
from confit import Cli


app = Cli(pretty_exceptions_show_locals=False)




@app.command(name="plots2")
def main(
    cohort_names: List[str],
    config_meta=None,
):
    conf_name =  config_meta["config_path"][0].stem
    conf = config_meta["resolved_config"]
    seed = conf["general"]["seed"]

    for cohort_name in cohort_names:
        dm = DataModule(conf)
        stays = dm.stays(cohort_name)
        cohort = dm.cohort(cohort_name)
        print(f"################################ {cohort_name} #######################")
        print(cohort_name, "size", len(cohort))

        if cohort_name == "all_population":
            cohort, stays = dm.sample(cohort, stays, n_1000=100, seed=seed)
        indicators = conf[cohort_name]["indicators"]

        print("indicators", indicators)

        for indicator in indicators:
            print("Indicator Class:",indicator.__class__)

            # Set conf dictionary
            conf_general = {"seed": seed, "cohort_name":cohort_name}

            
            if indicator.__class__ == Cluster:
                continue
                
            elif indicator.__class__ == Age:
                # Compute
                results = figureAge(cohort, stays, conf_general, variations_conf_table_1, indicator)
            
            else:      
                results = []
                for variation in variations_conf_table_1:
                    # Set conf dictionary
                    conf_general = {"seed": seed, "cohort_name":cohort_name}
                    conf_general.update(variation.copy())
                    
                    # Pseudonymize
                    dataset = get_pseudonimized_dataset(cohort, stays, **conf_general)
                    if variation["pseudonymization_algorithm"] == "NoPseudonymizer":
                        indicator_output = indicator.compute(dataset)
                    else:
                        indicator_output = indicator.compute(dataset, shifted=True)

                    conf_general.update(
                        dict(indicator_output=indicator_output, cohort_name=cohort_name)
                    )
                    results.append(conf_general.copy())


                

            filename = cohort_name + "/" + indicator.name
            _ = indicator.get_plot_class()(results, file_name=filename, conf_name=conf_name)

            # Another fig for Age difference
            if indicator.__class__ == Age:
                filename = cohort_name + "/" + indicator.name + "_diff"
                pc = indicator.get_plot_class()
                pc.compute_difference = True
                pc.ylabel = "Per-patient pseudonymisation-induced\ndifference of age (years)"
                _ = pc(results, file_name=filename, conf_name=conf_name)


if __name__ == "__main__":
    app()