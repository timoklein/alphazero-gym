from pathlib import Path
import pandas as pd
import yaml
import re
from typing import Union


def get_dataframe(path: Union[str, Path]) -> pd.DataFrame:

    dataset = []
    counter = 0

    for p_c, p_h in zip(
        Path(path).rglob("wandb/**/config.yaml"),
        log_dir.rglob("wandb/**/wandb-history.jsonl"),
    ):

        with open(p_c, "r") as f:
            c = pd.json_normalize(yaml.load(f, Loader=yaml.FullLoader))

        if c["Training episodes.value"][0] == 100:
            data = pd.read_json(p_h, lines=True)
            if len(data) == 100:
                c.drop(list(c.filter(regex=".desc")), axis=1, inplace=True)
                c["ID"] = counter
                c = c.rename(columns=lambda x: re.sub(".value", "", x))
                data = pd.read_json(p_h, lines=True)
                config_info = pd.concat([c] * len(data))

                config_info.reset_index(drop=True, inplace=True)
                data.reset_index(drop=True, inplace=True)

                run_data = pd.concat([config_info, data], axis=1)
                run_data.reset_index(drop=True, inplace=True)
                dataset.append(run_data)
                counter += 1

    runs = pd.concat(dataset)
    runs.reset_index(drop=True, inplace=True)
    return runs


if __name__ == "__main__":
    # get the outputs directory relative to this script
    dir_path = Path(__file__).resolve().parents[1]
    log_dir = dir_path / "outputs/"
    df = get_dataframe(path=log_dir)
    df.to_pickle(dir_path / "run_data.pickle")
