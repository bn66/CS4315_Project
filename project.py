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

from common import DATA_DIR, PLOT_DIR


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
    y_label_name: str = "airplane_type"
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
        xy_list: List[str] = self.x_feature_names + [self.y_label_name]

        # Read in data.
        df_xy: pd.DataFrame = pd.read_csv(prepared_dataset, usecols=xy_list)
        self.df_xy = df_xy.dropna()  # Remove NaN for now, worry later

        # Store X/Y Data Frames
        self.x_features: pd.DataFrame = self.df_xy.iloc[:, :-1]
        self.y_labels: pd.Series = self.df_xy[self.y_label_name]

        # Convert X/Y to arrays for input
        self.encoder: OneHotEncoder = OneHotEncoder()
        self.x_array: np.ndarray = self._create_x_array()
        self.y_array: np.ndarray = self.y_labels.to_numpy()

        # Create Names of columns
        self.x_names: List[str] = self._create_x_class_names()
        self.y_names: List[str] = np.unique(self.y_labels).tolist()

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
        print("~~~ Fitting model ~~~")
        self.fit_model()  # Fit Model and visualize tree
        print("~~~ Evaluating Model ~~~")
        self.evaluate_model()  # Test/Predict, visualize results
        # Save, Pickling?

    def savefig(self, ptype: str, **kwargs) -> None:
        """Save Figure."""
        plt.savefig(PLOT_DIR / f"{self.name}_{ptype}.png", **kwargs)
        plt.clf()

    @abstractmethod
    def define_model(self) -> None:
        """Define self.model."""

    @abstractmethod
    def fit_model(self) -> None:
        """Run model.fit(). Override with child class and add visualization."""
        self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self) -> None:
        """Use test set with model.predict and visualize results."""
        y_predict: np.ndarray = self.model.predict(self.x_test)

        # Use Confusion Matrix as visualization metric
        cf_mat: np.ndarray = confusion_matrix(self.y_test, y_predict)
        y_labels: np.ndarray = np.unique(np.hstack([self.y_test, y_predict]))
        disp: ConfusionMatrixDisplay = ConfusionMatrixDisplay(
            confusion_matrix=cf_mat, display_labels=y_labels
        )
        disp.plot(cmap=plt.cm.Greens)
        plt.xticks(rotation=90)
        plt.tight_layout()
        self.savefig("confusion_matrix")

        # Print out test for report.
        report: str = classification_report(
            self.y_test, y_predict, target_names=y_labels
        )
        tf_compare: np.ndarray = y_predict == self.y_test
        true_false: np.ndarray
        counts: np.ndarray
        true_false, counts = np.unique(tf_compare, return_counts=True)
        i: int
        tf: np.bool_  # pylint: ignore=invalid-name
        with open(PLOT_DIR / f"{self.name}_report.txt", "w", encoding="utf-8") as f_out:
            f_out.write(report)
            f_out.write(",".join([str(tf) for tf in true_false.tolist()]) + "\n")
            f_out.write(",".join([str(i) for i in counts.tolist()]) + "\n")
            for i, tf in enumerate(tf_compare):
                txt: str = ",".join(
                    [str(self.y_test[i]), str(y_predict[i]), str(tf), "\n"]
                )
                f_out.write(txt)

        breakpoint()
        # from sklearn.model_selection import cross_val_score

        # arr1, arr2 = np.unique(self.y_array, return_counts=True)
        # thing = {i: j for i, j in zip(arr1, arr2)}
        # cv_mse = np.mean(cross_val_score(model, self.x_array, self.y_array, cv=10))

        # print("Cross-validated MSE: {}".format(cv_mse))

        # return {'loss':cv_mse, 'status': STATUS_OK, 'model': model }


class DecisionTreeModel(Model):
    """Decision Tree CART Model."""

    name = "decision_tree"

    def define_model(self) -> None:
        self.model: tree.DecisionTreeClassifier = tree.DecisionTreeClassifier(
            min_samples_split=5, max_depth=10
        )

    def fit_model(self) -> None:
        """Fit and visualize."""
        super().fit_model()

        # Visualize
        tree.plot_tree(
            self.model,
            feature_names=self.x_names,
            class_names=self.y_names,
            impurity=True,
        )
        self.savefig("tree", dpi=2400)  # High res to see tree


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
    decision_tree.main()
