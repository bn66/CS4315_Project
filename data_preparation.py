"""Prepare data for model."""
from typing import List, Set
from pathlib import Path

import numpy as np
from numpy import radians, cos, sin, arcsin, sqrt
import pandas as pd

from common import DATA_DIR, TYPE_CODE_MAP, TYPE_CODE_MAP_SIMPLE


def lat_long_distance(
    lat1: np.ndarray, long1: np.ndarray, lat2: np.ndarray, long2: np.ndarray
) -> np.ndarray:
    """Calculate Distance between two points on lat/long

    Args:
        lat1: Numpy array of starting latitude, in decimal degrees
        long1: Numpy array of starting longitude, in decimal degrees
        lat2: Numpy array of end latitude, in decimal degrees
        long2: Numpy array of end longitude, in decimal degrees

    Returns:
        Result of Haversine formula for distance between two points in km,
            also known as the great circle distance.

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


def _add_airplane_type(df_data: pd.DataFrame) -> None:
    """Based on typecode, add airplane_type.

    Args:
        df_data: DataFrame of flight information.
    """
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


def _callsign_txt_only(df_data: pd.DataFrame) -> None:
    """Take the first three characters callsign, to identify airline.

    Args:
        df_data: DataFrame of flight information.
    """
    series_callsign: pd.Series = df_data["callsign"]
    callsign_txt: List[str] = []
    callsign: str
    digits: Set[str] = set(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    for callsign in series_callsign:
        char: str
        new_txt: str = "".join([char for char in callsign[:3] if char not in digits])
        if len(new_txt) < 3:
            new_txt = ""
        else:
            callsign_txt.append(new_txt)

    df_data["callsign_txt"] = callsign_txt


def _concat_origin_destination(df_data: pd.DataFrame) -> None:
    """Combine origin and destination names into one.

    Args:
        df_data: DataFrame of flight information.
    """
    # Put Origin and Destination Together
    array_route: pd.DataFrame = df_data[["origin", "destination"]].to_numpy()
    route: List[str] = []
    pair: np.ndarray
    for pair in array_route:
        from_to: str
        if pd.isna(pair).any():
            from_to = ""
        elif pair[0] == pair[1]:  # Origin and destination same, plane didn't move
            from_to = ""
        else:
            pair_list: List[str] = pair.tolist()
            pair_list.sort()
            from_to = "-".join(pair_list)

        route.append(from_to)

    df_data["route"] = route


def main(datapath: Path) -> None:
    """Prepare data for model.

    Adds the following:
        airplane type based off look-up table.
        reduce callsigns to first three letters
        calculate distance between origin and destination
        concatenate names of origin and destination


    Args:
        datapath: Path to data.

    Returns:
        Writes data to 'prepared_data.csv'
            and a truncated version 'prepared_data_short.csv'
    """
    df_data: pd.DataFrame = pd.read_csv(datapath)

    _add_airplane_type(df_data)
    _callsign_txt_only(df_data)
    _concat_origin_destination(df_data)

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
    df_data.iloc[:200].to_csv(DATA_DIR / "prepared_data_short.csv")


if __name__ == "__main__":
    dataset: Path = DATA_DIR / "flightlist_20190101_20190131.csv"
    # dataset: Path = DATA_DIR / "flightlist_short.csv"
    main(dataset)
