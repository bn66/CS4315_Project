"""Prepare data for model.
"""
from typing import List
from pathlib import Path

import pandas as pd

from common import DATA_DIR, TYPE_CODE_MAP


def main(datapath: Path) -> None:
    """Prepare data to model.

    Add airplane type based off look-up table.

    Args:
        datapath: Path to data.

    Returns:
        Writes data to 'prepared_data.csv'
    """
    df_data: pd.DataFrame = pd.read_csv(datapath)
    series_typecode: pd.Series = df_data["typecode"]
    airplane_types: List[str] = []
    typecode: str
    for typecode in series_typecode:
        value: str
        if typecode not in TYPE_CODE_MAP:
            value = ""
        else:
            value = TYPE_CODE_MAP[typecode]
        airplane_types.append(value)

    df_data["airplane_type"] = airplane_types
    df_data.to_csv(DATA_DIR / "prepared_data.csv")


if __name__ == "__main__":
    dataset: Path = DATA_DIR / "flightlist_20190101_20190131.csv"
    main(dataset)
