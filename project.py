"""CS 4315 Project.

"""
from abc import abstractmethod
from typing import List, Union, Optional, Any, Dict
from pathlib import Path

import numpy as np
from scipy.sparse import csc_matrix
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import ensemble
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    classification_report,
)
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials


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
        df_xy = df_xy.dropna()  # Remove NaN for now, worry later

        # Downsample due to potential memory issues
        df_x = df_xy.loc[:, self.x_feature_names]
        df_y = df_xy.loc[:, self.y_label_name]
        pct: float = 0.65  # Percentage to use in model
        self.df_xy: pd.DataFrame
        if pct < 1.00:
            _, x2, _, y2 = train_test_split(df_x, df_y, test_size=pct, stratify=df_y)
            print(f"Downsampling to {len(x2)}/{len(df_x)}: {pct:.2%} datapoints!")
            self.df_xy = pd.concat([x2, y2], axis=1)
        else:
            self.df_xy = df_xy

        # Store X/Y Data Frames
        self.x_features: pd.DataFrame = self.df_xy.iloc[:, :-1]
        self.y_labels: pd.Series = self.df_xy[self.y_label_name]

        # Convert X/Y to arrays for input
        self.encoder: OneHotEncoder = OneHotEncoder(sparse_output=True)
        self.x_array: np.ndarray = self._create_x_array()
        self.y_array: np.ndarray = self.y_labels.to_numpy()

        # Create Names of columns
        self.x_names: List[str] = self._create_x_class_names()
        self.y_names: List[str] = np.unique(self.y_labels).tolist()

        # Split into training and test sets
        x_train, x_test, y_train, y_test = train_test_split(
            self.x_array, self.y_array, test_size=0.10, stratify=self.y_array
        )
        self.x_train: np.ndarray = x_train
        self.x_test: np.ndarray = x_test
        self.y_train: np.ndarray = y_train
        self.y_test: np.ndarray = y_test

        self.y_predict: np.ndarray
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

    def main(self, **hyper_kwargs) -> None:
        """Main."""
        self.define_model(**hyper_kwargs)
        print("~~~ Fitting model ~~~")
        self.fit_model()  # Fit Model and visualize tree
        print("~~~ Evaluating model ~~~")
        self.evaluate_model()  # Test/Predict with model, set self.y_predict
        self.summarize_model()  # Summarize model
        # Save, Pickling, exporting?
        print("~~~ Done Model.main ~~~")

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
        if self.x_feature_strings:
            self.model.fit(csc_matrix(self.x_train), self.y_train)
        else:
            self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self) -> None:
        """Use test set with model.predict."""
        self.y_predict = self.model.predict(self.x_test)

    def summarize_model(self) -> None:
        """Visualize and confusion matrix and output a report."""
        # Use Confusion Matrix as visualization/metric
        cf_mat: np.ndarray = confusion_matrix(self.y_test, self.y_predict)
        np.savetxt(PLOT_DIR / f"{self.name}_confusion_matrix.txt", cf_mat)
        y_labels: np.ndarray = np.unique(np.hstack([self.y_test, self.y_predict]))
        disp: ConfusionMatrixDisplay = ConfusionMatrixDisplay.from_predictions(
            self.y_test,
            self.y_predict,
            labels=y_labels,
            xticks_rotation="vertical",
            cmap=mpl.colormaps["Greens"],
        )
        disp.figure_.set_figwidth(20)
        disp.figure_.set_figheight(20)
        self.savefig("confusion_matrix")

        # Write out test report.
        report: str = classification_report(
            self.y_test, self.y_predict, target_names=y_labels
        )
        tf_compare: np.ndarray = self.y_predict == self.y_test
        true_false: np.ndarray
        counts: np.ndarray
        true_false, counts = np.unique(tf_compare, return_counts=True)
        i: int
        tf: np.bool_  # pylint: disable=invalid-name
        with open(PLOT_DIR / f"{self.name}_report.txt", "w", encoding="utf-8") as f_out:
            f_out.write(report)
            f_out.write(",".join([str(tf) for tf in true_false.tolist()]) + "\n")
            f_out.write(",".join([str(i) for i in counts.tolist()]) + "\n")
            for i, tf in enumerate(tf_compare):  # pylint: disable=invalid-name
                txt: str = ",".join(
                    [str(self.y_test[i]), str(self.y_predict[i]), str(tf), "\n"]
                )
                f_out.write(txt)

            # Export Tree as Text
            if isinstance(self.model, tree.DecisionTreeClassifier):
                f_out.write("TREE EXPORT TEXT: \n")
                f_out.write(
                    tree.export_text(
                        self.model,
                        feature_names=self.x_names,
                        show_weights=True,
                    )
                )

            # Export Feature Importances
            f_out.write("Feature Importances: \n")
            f_out.write(",".join(self.x_names) + "\n")
            txt_list: List[float] = self.model.feature_importances_.tolist()
            f_out.write(",".join([str(i) for i in txt_list]) + "\n")
            f_out.write(txt + "\n")

            # Export Feature Importances
            f_out.write("Confusion matrix diagonal: \n")
            txt_list = y_labels.tolist()
            f_out.write(",".join([str(i) for i in txt_list]) + "\n")
            txt_list = cf_mat.diagonal().tolist()
            f_out.write(",".join([str(i) for i in txt_list]) + "\n")

            # Write out model info and hyper-parameters
            f_out.write(str(self.model) + "\n")

    def optimize(self, hp_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize model. Build model, evaluate and return hyperopt dict.

        Args:
            hp_kwargs: Hyper parameter kwarg dict passed to model constructor.

        Return:
            Hyper Opt Dictionary
        """

        print(f"Hyper-parameter combination: {hp_kwargs}")

        self.define_model(**hp_kwargs)

        # Use cross_val_score to optimize
        # score_fxn: None = None  # Default for DecisionTree is average 'accuracy'
        score_fxn: str = "f1_weighted"
        # score_fxn: str = "balanced_accuracy"  # Didn't give good results
        # Although cv=# defaults to StratifiedKFolds, I want to be explicit
        skf: StratifiedKFold = StratifiedKFold(n_splits=5)
        # If optimizing with hyperopt.fmin, we want the lowest score, multiply by -1
        cvs: np.ndarray = -1 * cross_val_score(
            self.model, self.x_train, self.y_train, cv=skf, scoring=score_fxn
        )

        std_dev: float = np.std(cvs)
        score: float = np.mean(cvs)
        print(f"Mean Score: {score} with Std. Dev. of {std_dev}")
        return {"loss": score, "status": STATUS_OK, "model": self.model}

    def run_optimizer(
        self, hyperspace: Dict[str, Any], run_main_after: bool = False
    ) -> None:
        """Optimize Hyper-Parameters."""
        trials: Trials = Trials()
        best = fmin(
            fn=self.optimize,
            space=hyperspace,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials,
        )

        losses: np.ndarray = np.array(trials.losses())
        plt.plot(losses)
        plt.title(f"Hyperopt loss minimization for {self.name}")
        plt.ylabel("Loss (f1_weighted)")
        plt.xlabel("Trial")
        self.savefig("optimization")
        # trials.result[0]  # Observe
        print(f"Hyperopt Best result!: {best}")

        if run_main_after:
            self.main(**best)


class DecisionTreeModel(Model):
    """Decision Tree CART Model."""

    name = "decision_tree"

    def define_model(self, **kwargs) -> None:
        """Set model and model parameters."""
        hyper_kwargs: Dict[str, Any] = {
            # criterion="gini",  # Default
            # splitter="best",  # Default
            "max_depth": 10,  # Max tree depth.
            "min_samples_split": 0.04762,  # Min number of samples for splitting
            # min_samples_leaf=1,  # Don't use if using weights
            "min_weight_fraction_leaf": 0.00953,  # Minimum weight in a leaf
            # max_features=None,  # Default
            # random_state=None,  # Default
            # max_leaf_nodes=None,  # Default
            # "min_impurity_decrease": 0.0,  # Didn't help in practice
            "class_weight": "balanced",  # Use
            # "ccp_alpha": 0.0,  # Didn't help in practice
        }

        k: str
        val: Any
        for k, val in kwargs.items():  # Update in args with correct type
            if k == "max_depth":
                hyper_kwargs[k] = int(val)
            elif k == "min_samples_split":
                hyper_kwargs[k] = float(val)
            elif k == "min_weight_fraction_leaf":
                hyper_kwargs[k] = float(val)
            elif k == "min_impurity_decrease":
                hyper_kwargs[k] = float(val)
            elif k == "ccp_alpha":
                hyper_kwargs[k] = float(val)

        self.model: tree.DecisionTreeClassifier = tree.DecisionTreeClassifier(
            **hyper_kwargs,
        )
        print(f"Model defined: {self.model}")

    def fit_model(self) -> None:
        """Fit and visualize."""
        super().fit_model()
        # print("Feature Importances: ")
        # print(self.model.feature_importances_)

        # Visualize
        tree.plot_tree(
            self.model,
            feature_names=self.x_names,
            class_names=self.y_names,
            impurity=True,
        )
        # fig: plt.figure = plt.gcf()
        # fig.set_figwidth(20)
        # fig.set_figheight(20)
        self.savefig("tree", dpi=2400)  # High res to see tree

    def set_then_run_optimizer(self, run_main_after: bool = False) -> None:
        """Set then run Hyper-Parameter Optimizer."""
        # fmt: off
        hyperspace: Dict[str, Any] = {
            "max_depth": hp.quniform("max_depth", 2, 10, 1),
            "min_weight_fraction_leaf": hp.uniform("min_weight_fraction_leaf", 0.0, 0.1),
            "min_samples_split": hp.uniform("min_samples_split", 0, 0.1),
            # "min_impurity_decrease": hp.uniform("min_impurity_decrease", 0.0, 0.4),  # Small knob in practice
            # "ccp_alpha": hp.uniform("ccp_alpha", 0, 0.5),  # Small knob in practice
        }

        self.run_optimizer(hyperspace, run_main_after)


class EnsembleForestModel(Model):
    """Ensemble Forest Models."""

    name = "ensemble_forest"

    def define_model(
        self,
        model: Union[ensemble.RandomForestClassifier, ensemble.ExtraTreesClassifier],
        **kwargs,
    ) -> Dict[str, Any]:
        """Set model and model parameters."""
        # Optimal RandomForest
        hyper_kwargs: Dict[str, Any] = {
            "n_estimators": 500,
            "max_depth": 9,
            "min_samples_split": 0.0936857,
            "min_weight_fraction_leaf": 0.0002781,
            "class_weight": "balanced",
        }

        # Optimal ExtremelyRandomTrees, TODO: Adjust code
        hyper_kwargs: Dict[str, Any] = {
            "n_estimators": 500,
            "max_depth": 5,
            "min_samples_split": 0.02398,
            "min_weight_fraction_leaf": 0.00041187,
            "class_weight": "balanced",
        }

        k: str
        val: Any
        for k, val in kwargs.items():
            if k == "max_depth":
                hyper_kwargs[k] = int(val)
            elif k == "min_samples_split":
                hyper_kwargs[k] = float(val)
            elif k == "min_weight_fraction_leaf":
                hyper_kwargs[k] = float(val)
            elif k == "n_estimators":
                hyper_kwargs[k] = int(val)

        self.model: Union[
            ensemble.RandomForestClassifier, ensemble.ExtraTreesClassifier
        ] = model(**hyper_kwargs)

        print(f"Model defined: {self.model}")
        print(f"Model n_estimators: {self.model.n_estimators}")

    def fit_model(self) -> None:
        """Fit and visualize."""
        super().fit_model()
        # print("Feature Importances: ")
        # print(self.model.feature_importances_)

        # Visualize
        print("Visualize tbd.")
        # Unsure how to visualize
        # self.savefig("ensemble", dpi=2400)  # High res to see

    def set_then_run_optimizer(self, run_main_after: bool = False) -> None:
        """Set then run Hyper-Parameters Optimizer."""
        # fmt: off
        hyperspace: Dict[str, Any] = {
            "max_depth": hp.quniform("max_depth", 2, 10, 1),
            "min_weight_fraction_leaf": hp.uniform("min_weight_fraction_leaf", 0.0, 0.1),
            "min_samples_split": hp.uniform("min_samples_split", 0, 0.1),
            # "min_impurity_decrease": hp.uniform("min_impurity_decrease", 0.0, 0.4),  # Small knob in practice
            # "ccp_alpha": hp.uniform("ccp_alpha", 0, 0.5),  # Small knob in practice
        }

        self.run_optimizer(hyperspace, run_main_after)


class RandomForestModel(EnsembleForestModel):
    """Random Forest Model."""

    name = "random_forest"

    def define_model(self, **kwargs) -> None:
        """Set model and model parameters."""
        model: ensemble.RandomForestClassifier = ensemble.RandomForestClassifier

        super().define_model(model, **kwargs)


class ExtremelyRandomTrees(EnsembleForestModel):
    """Random Forest Model."""

    name = "extremely_random_trees"

    def define_model(self, **kwargs) -> None:
        """Set model and model parameters."""
        model: ensemble.ExtraTreesClassifier = ensemble.ExtraTreesClassifier

        super().define_model(model, **kwargs)


if __name__ == "__main__":
    dataset: Path = DATA_DIR / "prepared_data.csv"
    # dataset: Path = DATA_DIR / "prepared_data_short.csv"

    decision_tree: DecisionTreeModel = DecisionTreeModel(dataset)
    decision_tree.set_then_run_optimizer(run_main_after=True)
    # decision_tree.main()

    random_forest: RandomForestModel = RandomForestModel(dataset)
    random_forest.set_then_run_optimizer(run_main_after=True)
    # # random_forest.main()

    extreme_random: ExtremelyRandomTrees = ExtremelyRandomTrees(dataset)
    extreme_random.set_then_run_optimizer(run_main_after=True)
    # # # extreme_random.main()
