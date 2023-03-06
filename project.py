"""CS 4315 Project.

"""
from typing import List, Set
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree

from common import DATA_DIR


class Model:
    """Model.

    Attributes:

    """

    y_name: str = "airplane_type"
    # y_name: str = "typecode"

    def __init__(self, prepared_dataset: Path):
        """Init.

        Args:
            prepared_dataset: Path to prepared data file.
        """
        self.df_full: pd.DataFrame = pd.read_csv(prepared_dataset, index_col=0)

        # Pick and choose features and predictor columns here
        self.x_feature_names = [
            "latitude_1",
            "longitude_1",
            "altitude_1",
            "latitude_2",
            "longitude_2",
            "altitude_2",
        ]
        xy_list: List[str] = self.x_feature_names + [self.y_name]  # predictor last
        col: str
        col_idx: List[int] = [self.df_full.columns.get_loc(col) for col in xy_list]

        self.df_xy: pd.DataFrame = self.df_full.iloc[:, col_idx]
        self.df_xy = self.df_xy.dropna()  # Remove NaN for now, worry about it later

        self.x_features: pd.DataFrame = self.df_xy.iloc[:, :-1]
        self.y_predict: pd.Series = self.df_xy[self.y_name]

        y_class_names: Set[str] = set(self.y_predict)
        self.y_class_names = list(y_class_names)
        self.y_class_names.sort()


class CARTModel(Model):
    """CART Model

    Take from W2 of Lectures.
    """

    def main(self) -> None:
        """Main."""

        model: tree.DecisionTreeClassifier = tree.DecisionTreeClassifier(max_depth=3)

        model.fit(self.x_features, self.y_predict)
        tree.plot_tree(
            model, feature_names=self.x_feature_names, class_names=self.y_class_names
        )
        plt.savefig("tree.jpg", dpi=2400)


if __name__ == "__main__":
    dataset: Path = DATA_DIR / "prepared_data.csv"

    ml_model: CARTModel = CARTModel(dataset)
    ml_model.main()
