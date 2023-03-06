"""Prepare data for model.

head -n 201 data/prepared_data.csv > data/prepared_data_short.csv 
"""
from typing import List, Set
from pathlib import Path

import numpy as np
from numpy import radians, cos, sin, arcsin, sqrt
import pandas as pd

from common import DATA_DIR, TYPE_CODE_MAP


# from math import radians, cos, sin, asin, sqrt
# def distance(lat1, lat2, lon1, lon2):
#     # The math module contains a function named
#     # radians which converts from degrees to radians.
#     lon1 = radians(lon1)
#     lon2 = radians(lon2)
#     lat1 = radians(lat1)
#     lat2 = radians(lat2)

#     # Haversine formula
#     dlon = lon2 - lon1
#     dlat = lat2 - lat1
#     a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2

#     c = 2 * asin(sqrt(a))

#     # Radius of earth in kilometers. Use 3956 for miles
#     r = 6371

#     # calculate the result
#     return c * r


# dist = distance(
#     lat1=-37.659485,
#     lon1=144.804421,
#     lat2=48.995316,
#     lon2=2.610802,
# )
# breakpoint()


def lat_long_distance(
    lat1: np.ndarray, long1: np.ndarray, lat2: np.ndarray, long2: np.ndarray
) -> np.ndarray:
    """Calculate Distance between two points on lat/long

    Args:
        ...

    Returns:
        Result of Haversine formula for distance between two points

    Sources:
        https://en.wikipedia.org/wiki/Great-circle_distance
        https://en.wikipedia.org/wiki/Haversine_formula
        https://www.geeksforgeeks.org/program-distance-two-points-earth/
    """

    # Convert all arrays from degrees into radians
    lat1 = radians(lat1)
    long1 = radians(long1)
    lat2 = radians(lat2)
    long2 = radians(long2)

    # Haversine formula
    dist_lat: np.ndarray = lat2 - lat1
    dist_long: np.ndarray = long2 - long1
    angle: np.ndarray = (
        sin(dist_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dist_long / 2) ** 2
    )

    dist: np.ndarray = 2 * arcsin(sqrt(angle))

    # Radius of earth in kilometers. Use 3956 for miles
    radius: float = 6371

    result: np.ndarray = dist * radius
    return result


def main(datapath: Path) -> None:
    """Prepare data to model.

    Add airplane type based off look-up table.

    Args:
        datapath: Path to data.

    Returns:
        Writes data to 'prepared_data.csv'
    """
    df_data: pd.DataFrame = pd.read_csv(datapath)

    # Based on typecode, add airplane_type
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

    # Split Call Sign into Text only
    series_callsign: pd.Series = df_data["callsign"]
    callsign_txt: List[str] = []
    callsign: str
    digits: Set[str] = set(["1", "2", "3", "4", "5", "6", "7", "8", "9"])
    for callsign in series_callsign:
        char: str
        new_txt: str = "".join([char for char in callsign if char not in digits])
        callsign_txt.append(new_txt)

    df_data["callsign_txt"] = callsign_txt

    # Calculate Distance
    distance: np.ndarray = lat_long_distance(
        lat1=df_data["latitude_1"],
        lat2=df_data["latitude_2"],
        long1=df_data["longitude_1"],
        long2=df_data["longitude_2"],
    )
    df_data["great_circle_distance"] = distance

    # Finally, output
    df_data.to_csv(DATA_DIR / "prepared_data.csv")


if __name__ == "__main__":
    dataset: Path = DATA_DIR / "flightlist_20190101_20190131.csv"
    main(dataset)
