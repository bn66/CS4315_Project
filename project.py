"""CS 4315 Project.

"""
from abc import abstractmethod
from typing import List, Union, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import ensemble
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    classification_report,
)

from common import DATA_DIR


class Model:
    """Model Parent Class.

    Model attributes are selected by commenting/uncommenting names

    Class Attributes:
        name: Model Name
        x_feature_floats: Names of Features that are numerical values
        x_feature_strings: Names of Features that are text-based
        x_feature_names: Combined names of all features
        y_predict_name: Name of variable to predict

    Attributes:
        # DataFrame objects
        df_xy: All Feature and Label data
        x_features: All Feature data
        y_predict: All Label data

        # np.array objects
        x_array: All Feature data as array
        y_array: All Label data as array

        encoder: One hot encoder for input information
        x_names: Names of each feature column in self.x_array
        y_names: Sorted List of unique output values in self.y_array

        # Split into training and test sets
        x_train, x_test, y_train, y_test = train_test_split(
            self.x_array, self.y_array, test_size=0.10
        )
        self.x_train: np.ndarray = x_train
        self.x_test: np.ndarray = x_test
        self.y_train: np.ndarray = y_train
        self.y_test: np.ndarray = y_test

        self.y_array_vectorized = [self.y_class_names_map[i] for i in self.y_train]
    """

    breakpoint()  ## Read only columns you want, read only data you want, dtypes speedup
    name: Optional[str] = None

    # Select x/features to use
    x_feature_floats: List[str] = ["great_circle_distance"]
    x_feature_strings: List[str] = [
        "callsign_txt",
        # "origin",
        # "destination",
        # "route",
    ]
    x_feature_names: List[str] = x_feature_floats + x_feature_strings

    # Select y/prediction
    y_predict_name: str = "airplane_type"
    # y_name: str = "typecode"

    def __init__(self, prepared_dataset: Path):
        """Initialize model.

        Args:
            prepared_dataset: Path to prepared data file.
        """
        self.model: Union[
            tree.DecisionTreeClassifier,
            ensemble.RandomForestClassifier,
            ensemble.ExtraTreesClassifier,
        ]
        # Read in all data.
        df_full: pd.DataFrame = pd.read_csv(prepared_dataset, index_col=0)

        # Make DataFrame only of data we need, with the predictor last
        xy_list: List[str] = self.x_feature_names + [self.y_predict_name]
        self.df_xy: pd.DataFrame = df_full.loc[:, xy_list]
        self.df_xy = self.df_xy.dropna()  # Remove NaN for now, worry later

        # Store X/Y Data Frames
        self.x_features: pd.DataFrame = self.df_xy.iloc[:, :-1]
        self.y_predict: pd.Series = self.df_xy[self.y_predict_name]

        # Convert X/Y to arrays for input
        self.encoder: OneHotEncoder = OneHotEncoder()
        self.x_array: np.ndarray = self._create_x_array()
        self.y_array: np.ndarray = self.y_predict.to_numpy()

        # Create Names of columns
        self.x_names: List[str] = self._create_x_class_names()
        self.y_names: List[str] = np.unique(self.y_predict).tolist()

        # Split into training and test sets
        x_train, x_test, y_train, y_test = train_test_split(
            self.x_array, self.y_array, test_size=0.10
        )
        self.x_train: np.ndarray = x_train
        self.x_test: np.ndarray = x_test
        self.y_train: np.ndarray = y_train
        self.y_test: np.ndarray = y_test

        print("Done Model.__init__!")

    def _create_x_array(self) -> np.ndarray:
        """Create input array.

        For strings, one-hot encode, for floats, simply add the column.
        If no float or string data, use only the other.

        Returns:
            input_array: Inputs in array format
        """
        # Process Float Information.
        df_floats: pd.DataFrame = self.df_xy.loc[:, self.x_feature_floats]
        arr_floats: np.ndarray
        if df_floats.empty:
            arr_floats = np.array([])
        else:
            arr_floats = df_floats.to_numpy()

        # Process String Information
        df_strings: pd.DataFrame = self.df_xy.loc[:, self.x_feature_strings]
        arr_strings: np.ndarray
        if df_strings.empty:
            arr_strings = np.array([])
        else:
            array_strings = df_strings.to_numpy()
            self.encoder.fit(array_strings)
            # print(f"self.encoder categories: {self.encoder.categories_}")
            arr_strings = self.encoder.transform(array_strings).toarray()

        input_array: np.ndarray
        if df_strings.empty:
            input_array = arr_floats
        elif df_floats.empty:
            input_array = arr_strings
        else:
            input_array = np.hstack([arr_floats, arr_strings])
        return input_array

    def _create_x_class_names(self) -> List[str]:
        """Create list of x class names.

        Returns:
            x_class_names: List of x class names
        """
        float_names: List[str] = self.x_feature_floats

        if self.x_feature_strings:
            string_names: List[str] = np.hstack(self.encoder.categories_).tolist()

        x_class_names: List[str] = float_names + string_names
        return x_class_names

    def main(self) -> None:
        """Main."""
        self.define_model()  # Define Model to be used
        self.fit_model()  # Fit Model
        # Post-visualize data
        # Save

    @abstractmethod
    def define_model(self) -> None:
        """Define self.model."""

    @abstractmethod
    def fit_model(self) -> None:
        """Run model.fit()."""


class DecisionTreeModel(Model):
    """Decision Tree CART Model."""

    name = "decision_tree"

    def define_model(self) -> None:
        self.model: tree.DecisionTreeClassifier = tree.DecisionTreeClassifier(
            min_samples_split=5, max_depth=10
        )

    def fit_model(self) -> None:
        """Fit the model."""
        # from sklearn.model_selection import cross_val_score

        # arr1, arr2 = np.unique(self.y_array, return_counts=True)
        # thing = {i: j for i, j in zip(arr1, arr2)}
        # cv_mse = np.mean(cross_val_score(model, self.x_array, self.y_array, cv=10))

        # print("Cross-validated MSE: {}".format(cv_mse))

        # return {'loss':cv_mse, 'status': STATUS_OK, 'model': model }
        breakpoint()

        self.model.fit(self.x_train, self.y_train)
        y_predict = self.model.predict(self.x_test)
        yt_unique = np.unique(np.hstack([self.y_test, y_predict]))
        bool_compare = y_predict == self.y_test
        # accuracy = np.unique(cool_compare, return_counts=True)

        for i in range(0, len(bool_compare)):
            print(self.y_test[i], y_predict[i], bool_compare[i])
        # Confusion Matrix, for accuracy
        breakpoint()
        cm = confusion_matrix(self.y_test, y_predict)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=yt_unique)
        disp.plot(cmap=plt.cm.Greens)
        plt.xticks(rotation=90)
        plt.savefig("decision_tree_confusion_matrix.jpg", dpi=2400)

        print(classification_report(self.y_test, y_predict, target_names=yt_unique))

        breakpoint()

        # Visualize
        # tree.plot_tree(
        #     model,
        #     feature_names=self.x_class_names,
        #     class_names=self.y_class_names,
        #     impurity=True,
        # )
        # plt.savefig("decision_tree_scatter.jpg", dpi=2400)

        # # Visualize as scatter plot
        # constant_y: np.ndarray = np.zeros((self.x_train.shape[0]))
        # fig, ax = plt.subplots()
        # for name in self.y_class_names:
        #     idx = np.where(name == self.y_train)

        #     ax.scatter(
        #         self.x_train[idx, 0],
        #         constant_y[idx],
        #         c=self.c_dict_map[name],
        #         cmap=plt.cm.Set1,
        #         edgecolor="k",
        #     )

        # ax.legend(
        #     self.y_class_names,
        #     loc="upper center",
        #     bbox_to_anchor=(0.5, 0.3),
        #     fancybox=True,
        #     ncol=2,
        # )

        # plt.savefig("decision_tree_scatter.jpg", dpi=2400)
        # breakpoint()


class RandomForestModel(Model):
    """Random Forest Model."""

    def main(self) -> None:
        """Main."""

        model: ensemble.RandomForestClassifier = ensemble.RandomForestClassifier(
            min_samples_split=5, max_depth=10
        )


class ExtremelyRandomTrees(Model):
    """Random Forest Model."""

    def main(self) -> None:
        """Main."""

        model: ensemble.ExtraTreesClassifier = ensemble.ExtraTreesClassifier(
            min_samples_split=5, max_depth=10
        )


if __name__ == "__main__":
    # dataset: Path = DATA_DIR / "prepared_data.csv"
    dataset: Path = DATA_DIR / "prepared_data_short.csv"

    decision_tree: DecisionTreeModel = DecisionTreeModel(dataset)
    decision_tree.visualize_input_data()
    decision_tree.main()
