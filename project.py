"""CS 4315 Project.

"""
from typing import List, Set
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder

from common import DATA_DIR


class Model:
    """Model.

    Attributes:

    """

    # Select x/features to use
    x_feature_floats: List[str] = ["great_circle_distance"]
    x_feature_strings: List[str] = [
        "callsign_txt",
        "origin",
        "destination",
        "route",
    ]
    x_feature_names: List[str] = x_feature_floats + x_feature_strings

    # Select y/prediction
    y_predict_name: str = "airplane_type"
    # y_name: str = "typecode"

    def __init__(self, prepared_dataset: Path):
        """Init.

        Args:
            prepared_dataset: Path to prepared data file.
        """
        self.df_full: pd.DataFrame = pd.read_csv(prepared_dataset, index_col=0)

        # Make DataFrame only of data we need, with the predictor last
        xy_list: List[str] = self.x_feature_names + [self.y_predict_name]
        self.df_xy: pd.DataFrame = self.df_full.loc[:100, xy_list]
        self.df_xy = self.df_xy.dropna()  # Remove NaN for now, worry later

        # Store X/Y Data Frames
        self.x_features: pd.DataFrame = self.df_xy.iloc[:, :-1]
        self.y_predict: pd.Series = self.df_xy[self.y_predict_name]

        # Convert X/Y to arrays for input
        self.encoder: OneHotEncoder = OneHotEncoder()
        self.x_array: np.ndarray = self._create_x_array()
        self.y_array: np.ndarray = self.y_predict.to_numpy()

        # Things for Post-Processing
        self.x_class_names: List[str] = self._create_x_class_names()
        y_class_names: Set[str] = set(self.y_predict)
        self.y_class_names = list(y_class_names)
        self.y_class_names.sort()
        print("Done Model.__init__!")

        # Consider doing Assertion Checks

    def _create_x_array(self) -> np.ndarray:
        """Create input array.

        For strings, one-hot encode, for floats, simply add the column.

        Returns:
            input_array: Inputs in array format
        """
        # Process Float Information
        df_floats: pd.DataFrame = self.df_xy.loc[:, self.x_feature_floats]
        # print(f"self.encoder categories: {self.encoder.categories_}")
        arr_floats: np.ndarray = df_floats.to_numpy()

        # Process String Information
        df_strings: pd.DataFrame = self.df_xy.loc[:, self.x_feature_strings]
        array_strings: np.ndarray = df_strings.to_numpy()
        self.encoder.fit(array_strings)
        # print(f"self.encoder categories: {self.encoder.categories_}")
        arr_strings: np.ndarray = self.encoder.transform(array_strings).toarray()

        input_array: np.ndarray = np.hstack([arr_floats, arr_strings])
        return input_array

    def _create_x_class_names(self) -> List[str]:
        """Create list of x class names.

        Returns:
            x_class_names: List of x class names
        """
        float_names: List[str] = self.x_feature_floats
        string_names: List[str] = np.hstack(self.encoder.categories_).tolist()

        x_class_names: List[str] = float_names + string_names
        return x_class_names


class CARTModel(Model):
    """CART Model

    Take from W2 of Lectures.
    """

    def main(self) -> None:
        """Main."""

        model: tree.DecisionTreeClassifier = tree.DecisionTreeClassifier(
            min_samples_split=5, max_depth=10
        )

        model.fit(self.x_array, self.y_array)
        tree.plot_tree(
            model,
            feature_names=self.x_class_names,
            class_names=self.y_class_names,
            impurity=True,
        )
        plt.savefig("tree.jpg", dpi=2400)


if __name__ == "__main__":
    # dataset: Path = DATA_DIR / "prepared_data.csv"
    dataset: Path = DATA_DIR / "prepared_data_short.csv"

    ml_model: CARTModel = CARTModel(dataset)
    ml_model.main()
