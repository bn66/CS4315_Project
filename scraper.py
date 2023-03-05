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


class Scraper:
    """Scraper.

    Attributes:
        self.icao_list: Array of icaos
        self.completed_icao: Completed ICAOs
        self.df_old: Optional Data Frame of old output, for continuation
        self.df_old_completed: Set of completed searches
        self.errors: Old Errors
        self.data_dict: Dictionary of with Column names mapped to indices for
            the table that is to be parsed.
    """

    url: str = "https://www.planespotters.net/hex/{}"  # Bypass bot
    # http://webcache.googleusercontent.com/search?q=cache:https://www.planespotters.net/hex/888152
    # url: str = 'https://opensky-network.org/aircraft-profile?icao24={}'  # No Plane information
    # url: str = 'http://www.airframes.org/'  # No Plane information
    # https://www.airport-data.com/aircraft/N900GX.html
    output_file: Path = DATA_DIR / "scraped_data.csv"
    output_icao_key: str = "ICAO24"
    error_file: Path = DATA_DIR / "scraped_data_errors.txt"

    def __init__(self, icao_path: Path):
        """Init.

        Args:
            icao_path: Path to a file with unique ICAOs
        """
        self.icao_list: np.ndarray = np.loadtxt(icao_path, dtype=str)
        self.completed_icao: Set[str] = set()

        # Read in old data for continuing
        self.df_old: Optional[pd.DataFrame] = self.read_old_output_file()
        self.df_old_completed: Set[str] = set()
        if self.df_old is not None:
            self.df_old_completed = set(self.df_old[self.output_icao_key])

        self.errors: Set[str] = self.read_old_error_file()

        # data_dict["HEADER"] = (column_index, data_list)
        self.data_dict: Dict[str, Tuple[int, List[str]]] = {
            "REGISTRATION": tuple([2, []]),
            "MANUFACTURER_SERIAL_NUMBER": tuple([3, []]),
            "AIRCRAFT_TYPE": tuple([4, []]),
            "AIRLINE": tuple([5, []]),
            "DELIVERED": tuple([6, []]),
            "STATUS": tuple([7, []]),
            "PREVIOUS_REGISTRATION": tuple([8, []]),
            # "REMARK": tuple(9, []),
        }

    def main(self) -> None:
        """Main."""
        breakpoint()
        options = Options()
        # options.add_argument('--headless')
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=options
        )

        i: int
        icao24: str
        x_path_base: str = '//*[@id="content"]/div/div[2]/div[2]/div[{}]'
        # Scan "Current Registration Records table"
        no_element_found: bool = False
        for i, icao24 in enumerate(self.data_df["icao24"]):
            if i > 0:  # Back up files every 1000
                if (i % 1000) == 0:
                    print(f"Backing up files! i={i}")
                    self.export_data_dict()

            if icao24 in self.df_old_completed:
                print(f"{icao24} found in old file! Skipping!")
                continue
            elif icao24 in self.completed_icao:
                print(f"{icao24} already found! Skipping!")
                continue
            elif icao24 in self.errors:
                print(f"{icao24} was an error! Skipping!")
                continue

            self.completed_icao[icao24] = None
            driver.get(self.url.format(icao24.upper()))

            # Check if Captcha
            if driver.title == "Please verify your request":
                print("CAPTCHA DETECTED, PLEASE CHECK BROWSER!")
                breakpoint()
            elif driver.title == "Login required":
                print("Oh hey you mined too much data! Please login.")
                breakpoint()
            elif driver.title == "Data Limit Reached":
                print("Oh hey you mined too much data! Should stop.")
                breakpoint()

            values: Tuple[int, List[str]]
            for values in self.data_dict.values():
                # Unpack Values
                col: int = values[0]
                data_list: List[str] = values[1]

                # Mine Data
                x_path: str = x_path_base.format(col)
                try:
                    element: WebElement = driver.find_element(By.XPATH, x_path)
                except NoSuchElementException:
                    print(f"NoSuchElementException for {icao24}!")
                    self.errors.add(icao24)
                    no_element_found = True
                    break

                value: str = element.text
                data_list.append(value)
                # print(col, value)

            if no_element_found:
                no_element_found = False
                continue

        # Finally, Export Data
        breakpoint()
        self.export_data_dict()

        # Export errors
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
        icao: str
        for icao in self.errors:
            del self.completed_icao[icao]
        new_dict[self.output_icao_key] = list(self.completed_icao.keys())

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
        if self.error_file.exists():
            error_set: Set[str] = set()
            with open(self.error_file, "r", encoding="utf-8") as f_in:
                line: str
                for line in f_in:
                    error_set.add(line.rstrip())
            return error_set
        else:
            return set()


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

    icao_list: Path = unique_icao_from_dataset(dataset)

    scraper: Scraper = Scraper(icao_list)
    scraper.main()
