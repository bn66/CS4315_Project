"""Prepare and visualize data for model."""
from typing import List, Set
from pathlib import Path

import numpy as np
from numpy import radians, cos, sin, arcsin, sqrt
import pandas as pd
import matplotlib.pyplot as plt

from common import (
    DATA_DIR,
    PLOT_DIR,
    TYPE_CODE_MAP,
    TYPE_CODE_MAP_SIMPLE,
    PANDAS_DTYPE,
    PANDAS_DATETIME,
)

DIGITS: Set[str] = set(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])


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
        if typecode not in TYPE_CODE_MAP_SIMPLE:
            value = ""
        else:
            value = TYPE_CODE_MAP_SIMPLE[typecode]
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
    for callsign in series_callsign:
        char: str
        new_txt: str = "".join([char for char in callsign[:3] if char not in DIGITS])

        if len(new_txt) < 3:
            new_txt = ""

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
        char: str
        if pd.isna(pair).any():
            from_to = ""
        elif pair[0] == pair[1]:  # Same Origin and Destination
            from_to = ""
        elif any(char in DIGITS for char in pair[0]):  # Origin not all letters
            from_to = ""
        elif any(char in DIGITS for char in pair[1]):  # Dest. not all letters
            from_to = ""
        else:
            pair_list: List[str] = pair.tolist()
            pair_list.sort()
            from_to = "-".join(pair_list)

        route.append(from_to)

    df_data["route"] = route


def _calculate_distance(df_data: pd.DataFrame) -> None:
    """Calculate distance between points. Also ignore arbitrarily low values.

    Args:
        df_data: DataFrame of flight information.
    """
    # Calculate Distance
    distance: np.ndarray = lat_long_distance(
        lat1=df_data["latitude_1"],
        lat2=df_data["latitude_2"],
        long1=df_data["longitude_1"],
        long2=df_data["longitude_2"],
    )

    # Ignore all flights below 500 km
    index: np.ndarray = np.where(distance < 500)[0]
    distance[index] = np.nan

    df_data["great_circle_distance"] = distance


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
    _calculate_distance(df_data)

    # Finally, output
    df_data.to_csv(DATA_DIR / "prepared_data.csv")
    df_data.iloc[:200].to_csv(DATA_DIR / "prepared_data_short.csv")


class Visualize:
    """Visualize class."""

    def __init__(self, prep_dataset: Path):
        """Initialize.

        Args:
            prep_dataset: Prepared data set output from main()
        """
        self.name_stem: str = "input_data"
        df_data: pd.DataFrame = pd.read_csv(
            prep_dataset, index_col=0, dtype=PANDAS_DTYPE, parse_dates=PANDAS_DATETIME
        )

        # Remove non-sense values
        df_data = df_data.drop("number", axis=1)  # Column has only NaN
        self.df_data: pd.DataFrame = df_data.dropna()

        # Common y-variables
        self.y_name: str = "airplane_type"

    def plot_scatter_y_vs_gcm(self) -> None:
        """Scatter Plot the y-label vs. great circle distance.

        Returns:
            Saves a plot
        """
        y_df: pd.DataFrame = self.df_data[self.y_name]
        y_names: np.ndarray = np.unique(self.df_data[self.y_name])

        _, axs = plt.subplots(2, 1)
        col_idx: int = self.df_data.columns.get_loc("great_circle_distance")
        constant_y: np.ndarray = np.zeros((self.df_data.shape[0]))

        # Plot points with unique colors
        cmap = plt.get_cmap("gist_rainbow")
        axs[0].set_prop_cycle(color=[cmap(1.0 * i / 20) for i in range(20)])

        # For each y_name plot their points
        y_name: str
        for y_name in y_names:
            idx: np.ndarray = np.where(y_name == y_df)[0]

            axs[0].scatter(
                self.df_data.iloc[idx, col_idx],
                constant_y[idx],
            )

        axs[0].legend(y_names, loc=(-0.1, -1.5), fancybox=True, ncol=2)
        axs[1].axis("off")
        axs[0].set_yticks(())  # Turn off
        plt.savefig(
            PLOT_DIR / f"{self.name_stem}_scatter_label_vs_distance.jpg", dpi=2400
        )

    def plot_histograms_all(self) -> None:
        """Plot histograms for all columns.

        Returns:
            Saves a plot
        """

        # List of columns to histogram
        col_list: List[str] = [
            "great_circle_distance",
            "callsign_txt",
            "origin",
            "destination",
            "route",
            "airplane_type",
        ]
        col_name: str

        header: List[str] = []
        for col_name in col_list:
            header.append(f"{col_name}_bins")
            header.append(f"{col_name}_counts")

        arrays_out: List[np.ndarray] = []
        for col_name in col_list:
            print(col_name)
            data: pd.Series = self.df_data[col_name]

            if data.dtype == np.float64:
                counts, bins, _ = plt.hist(data, bins=10)
            else:  # Assuming string
                # Because of slowdown at large scale, do manually with np
                bins, counts = np.unique(data, return_counts=True)
                plt.bar(bins, counts)

            arrays_out.append(bins)
            arrays_out.append(counts)

            plt.title(f"Histogram of {col_name}")
            plt.ylabel("count")
            plt.xticks(rotation=90)
            plt.xlabel(col_name)
            plt.tight_layout()
            plt.savefig(PLOT_DIR / f"histogram_{col_name}.jpg")
            plt.clf()

        df_out: pd.DataFrame = pd.DataFrame(arrays_out)
        df_out = df_out.T
        df_out.columns = header
        df_out.to_csv(PLOT_DIR / "histogram_counts.csv")

    def main(self) -> None:
        """Visualize DataSet Data."""
        # # Plot great circle distances on scatter plot
        # self.plot_scatter_y_vs_gcm()

        # Plot histograms of all columns
        self.plot_histograms_all()

        breakpoint()


if __name__ == "__main__":
    # dataset: Path = DATA_DIR / "flightlist_20190101_20190131.csv"
    # # dataset: Path = DATA_DIR / "flightlist_short.csv"
    # main(dataset)

    prepared_dataset: Path = DATA_DIR / "prepared_data.csv"
    # prepared_dataset: Path = DATA_DIR / "prepared_data_short.csv"
    visualize: Visualize = Visualize(prepared_dataset)
    visualize.main()
