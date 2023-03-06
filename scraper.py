"""Data Scraper to get aircraft type from icao24

TODO:
    Speed up through parallelization? (multithread might be best)
    Headless? Auto-put in passsword?
    Request with login?
    Feed only unique ID's
"""
# from future import annotations
from typing import TYPE_CHECKING
from typing import Dict, Tuple, List, Set, Optional
from pathlib import Path
import threading

import numpy as np
import pandas as pd

# import requests
# from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

from common import DATA_DIR

if TYPE_CHECKING:
    from selenium.webdriver.remote.webelement import WebElement


class ScrapeThread(threading.Thread):
    """Scraper thread for multi-threading."""

    url: str = "https://www.planespotters.net/hex/{}"
    # url: str = "http://webcache.googleusercontent.com/search?q=cache:https://www.planespotters.net/hex/{}"

    def __init__(self, icao_list: Set[str], output_path: Path) -> None:
        """Init.

        Args:
            icao_list: ICAOs to iterate over in this thead.
        """
        super().__init__()
        self.icao_list: Set[str] = icao_list
        self.completed_icao: Set[str] = set()

        self.output_file: Path = output_path
        self.errors: Set[str] = set()

        # data_dict["HEADER"] = (column_index, data_list)
        self.data_dict: Dict[str, Tuple[int, List[str]]] = {
            "REGISTRATION": tuple([2, []]),  # type: ignore
            "MANUFACTURER_SERIAL_NUMBER": tuple([3, []]),  # type: ignore
            "AIRCRAFT_TYPE": tuple([4, []]),  # type: ignore
            "AIRLINE": tuple([5, []]),  # type: ignore
            "DELIVERED": tuple([6, []]),  # type: ignore
            "STATUS": tuple([7, []]),  # type: ignore
            "PREVIOUS_REGISTRATION": tuple([8, []]),  # type: ignore
            # "REMARK": tuple(9, []),
        }

    def run(self) -> None:
        """Run Scraper Thread."""
        options: Options = Options()
        # options.add_argument('--headless')
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=options
        )

        i: int
        icao24: str
        x_path_base: str = '//*[@id="content"]/div/div[2]/div[2]/div[{}]'
        # Scan "Current Registration Records table"
        for i, icao24 in enumerate(self.icao_list):
            if i > 0:  # Back up files every 1000
                if (i % 1000) == 0:
                    print(f"At #{i}")
            #         print(f"Backing up files! i={i}")
            #         self.export_data_dict()

            self.completed_icao.add(icao24)

            driver.get(self.url.format(icao24.upper()))

            # Check on webpage title to see if correct webpage
            correct_title: str = f"{icao24.upper()} Mode S Code | Aircraft Data"
            if driver.title == "Please verify your request":
                print("CAPTCHA DETECTED, PLEASE CHECK BROWSER!")
                breakpoint()
            elif driver.title == "Login required":
                print("Oh hey you mined too much data! Please login.")
                breakpoint()
            elif driver.title == "Data Limit Reached":
                print("Oh hey you mined too much data! Should stop.")
                breakpoint()
            elif driver.title != correct_title:
                print("Unexpected webpage happened!")
                breakpoint()

            # For a given webpage, iterate over the table
            values: Tuple[int, List[str]]
            for values in self.data_dict.values():
                # Unpack Values
                col: int = values[0]
                data_list: List[str] = values[1]

                # Mine Data
                x_path: str = x_path_base.format(col)
                value: str
                try:
                    element: WebElement = driver.find_element(By.XPATH, x_path)
                    value = element.text
                except NoSuchElementException:
                    # print(f"NoSuchElementException for {icao24} on col {col}")
                    self.errors.add(icao24)
                    value = ""

                # Finally, add stuff to table
                # print(col, value)
                data_list.append(value)

        self.data_dict["ICAO24"] = list(self.completed_icao)


class Scraper:
    """Scraper.

    Attributes:
        self.icao_list: Array of icaos imported from a file
        self.completed_icao: Completed ICAOs
        self.df_old: Optional Data Frame of old output, for continuation
        self.completed_old: Set of completed searches
        self.errors: Old Errors
        self.data_dict: Dictionary of with Column names mapped to indices for
            the table that is to be parsed.
    """

    # url: str = 'https://opensky-network.org/aircraft-profile?icao24={}'  # No Plane information
    # url: str = 'http://www.airframes.org/'  # No Plane information
    # https://www.airport-data.com/aircraft/N900GX.html
    output_file: Path = DATA_DIR / "scraped_data.csv"
    output_icao_key: str = "ICAO24"
    error_file: Path = DATA_DIR / "scraped_data_errors.txt"

    def __init__(self, icao_path: Path, threads: int = 1):
        """Init.

        Args:
            icao_path: Path to a file with unique ICAOs
            threads: Number of threads to use for multithreading
        """
        self.icao_list: Set[str] = set(np.loadtxt(icao_path, dtype=str))
        self.completed_icao: Set[str] = set()
        self.threads: int = threads

        # Read in old data for continuing
        self.df_old: Optional[pd.DataFrame] = self.read_old_output_file()
        self.completed_old: Set[str] = set()
        if self.df_old is not None:
            self.completed_old = set(self.df_old[self.output_icao_key])

        self.errors: Set[str] = self.read_old_error_file()

        # Remove already completed entries
        print(
            f"Started with {len(self.icao_list)} icao entries, removing ",
            f"{len(self.completed_old)} entries from old output and",
            f"{len(self.errors)} entries from error file",
        )
        self.icao_list -= self.completed_old
        self.icao_list -= self.errors
        print(f"New icao length: {len(self.icao_list)}")

        # data_dict["HEADER"] = (column_index, data_list)
        self.data_dict: Dict[str, Tuple[int, List[str]]] = {
            "REGISTRATION": tuple([2, []]),  # type: ignore
            "MANUFACTURER_SERIAL_NUMBER": tuple([3, []]),  # type: ignore
            "AIRCRAFT_TYPE": tuple([4, []]),  # type: ignore
            "AIRLINE": tuple([5, []]),  # type: ignore
            "DELIVERED": tuple([6, []]),  # type: ignore
            "STATUS": tuple([7, []]),  # type: ignore
            "PREVIOUS_REGISTRATION": tuple([8, []]),  # type: ignore
            # "REMARK": tuple(9, []),
        }

    def run(self) -> None:
        """Run."""
        # Split ICAO ids into n-even sets.
        icao_partitions: List[np.ndarray] = np.array_split(
            np.array(list(self.icao_list), dtype=str), self.threads
        )

        threads: List[ScrapeThread] = []
        i: int
        partition: np.ndarray
        for i, partition in enumerate(icao_partitions):
            icao_list: List[str] = set(partition.tolist())
            path: Path = self.output_file.parent
            output_name: str = (
                f"{self.output_file.stem}_thread{i+1}{self.output_file.suffix}"
            )
            full_output_path: Path = path / output_name
            thread: ScrapeThread = ScrapeThread(icao_list, full_output_path)
            thread.start()
            threads.append(thread)

        # Wait for all threads to finish.
        for thread in threads:
            thread.join()

        # Combine and output thread data_dict, need to break this into function
        df_list: List[pd.DataFrame] = []
        for thread in threads:
            # Create a dictionary only with 'ColumnName' to DataValues
            key: str
            value: Tuple[int, List[str]]
            new_dict: Dict[str, List[str]] = {
                key: value[1] for key, value in thread.data_dict.items()
            }
            df_thread: pd.DataFrame = pd.DataFrame.from_dict(new_dict)
            df_list.append(df_thread)

        df_out: pd.DataFrame = pd.concat(df_list)
        df_out.to_csv(self.output_file)

        # Combine errors from each thread and output
        for thread in threads:
            self.errors.update(thread.errors)

        self.export_errors()

    def export_data_dict(self) -> None:
        """Export data dictionary. Add ICAOs and append to old file."""
        key: str
        value: Tuple[int, List[str]]
        new_dict: Dict[str, List[str]] = {
            key: value[1] for key, value in self.data_dict.items()
        }
        # df_padded = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in new_dict.items() ]))

        # Remove Error keys
        new_dict[self.output_icao_key] = list(self.completed_icao)

        df_new: pd.DataFrame = pd.DataFrame.from_dict(new_dict)
        df_out: pd.DataFrame = pd.concat([self.df_old, df_new])
        df_out.to_csv(self.output_file)
        print(f"Exported data to {self.output_file}!")

    def export_errors(self) -> None:
        """Export errors."""
        with open(self.error_file, "a", encoding="utf-8") as f_out:
            out_string: str = "\n".join(self.errors)
            f_out.write(out_string)

        print(f"Errors appended at: {self.error_file}")

    def read_old_output_file(self) -> Optional[pd.DataFrame]:
        """Read the old output file."""
        if self.output_file.exists():
            df_old: pd.DataFrame = pd.read_csv(str(self.output_file), index_col=0)
            return df_old
        else:
            return None

    def read_old_error_file(self) -> Set[str]:
        """Read the old error file."""
        error_set: Set[str] = set()
        if self.error_file.exists():
            error_set = set(np.loadtxt(self.error_file, dtype=str))

        return error_set


def unique_icao_from_dataset(datapath: Path) -> Path:
    """Generate list of unique ICAO24 to iterate on based on dataset.

    Args:
        datapath: Path to dataset

    Returns:
        output_file: Path to list of unique ICAOs
    """
    data_df: pd.DataFrame = pd.read_csv(str(datapath))

    output_file: Path = Path(DATA_DIR / "unique_icao_list.txt")
    icao_set: Set[str] = set(data_df["icao24"])
    with open(output_file, "w", encoding="utf-8") as f_out:
        icao: str
        for icao in icao_set:
            f_out.write(f"{icao}\n")

    return output_file


if __name__ == "__main__":
    dataset: Path = DATA_DIR / "flightlist_20190101_20190131.csv"
    # dataset_test: Path = DATA_DIR / "flightlist_short.csv"

    icao_set: Path = unique_icao_from_dataset(dataset)
    # icao_set: Path = DATA_DIR / "unique_icao_list_test.txt"

    scraper: Scraper = Scraper(icao_set, threads=4)
    scraper.run()
