from confit import Cli
from typing import List
from privacy import registry

app = Cli(pretty_exceptions_show_locals=False)



@app.command(name="plots")
def main(
    figures_functions: List[str],
    config_meta=None,
):
    conf_name = config_meta["config_path"][0].stem
    conf = config_meta["resolved_config"]

    for f_name in figures_functions:
        f = registry.plots.get(f_name)
        f(**dict(conf=conf, conf_name=conf_name))


if __name__ == "__main__":
    app()
