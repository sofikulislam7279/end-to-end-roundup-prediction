import sys
import importlib
from typing import Tuple

import numpy as np

from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)

from roundup.exception import RoundupException
from roundup.logger import logging

from roundup.utils.main_utils import (
    load_numpy_array_data,
    load_object,
    save_object,
    read_yaml_file
)

from roundup.entity.config_entity import ModelTrainerConfig

from roundup.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    RegressionMetricArtifact
)

from roundup.entity.estimator import RoundupModel


class ModelTrainer:

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig
    ):
        """
        Method Name : __init__

        Description :
            Initializes ModelTrainer with data transformation
            artifact and model trainer configuration.

        On Failure :
            Raises RoundupException.
        """

        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config

        except Exception as e:
            raise RoundupException(e, sys) from e

    @staticmethod
    def _import_class(
        module_name: str,
        class_name: str
    ):
        """
        Method Name : _import_class

        Description :
            Dynamically imports a class from the specified module.

        Output :
            Returns imported class.

        On Failure :
            Raises RoundupException.
        """

        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)

        except Exception as e:
            raise RoundupException(e, sys) from e
        

    def get_model_object_and_report(
        self,
        train: np.ndarray,
        test: np.ndarray
    ) -> Tuple[object, RegressionMetricArtifact]:

        """
        Method Name : get_model_object_and_report

        Description :
            Trains candidate regression models, performs
            RandomizedSearchCV for tunable models, selects the
            best model based on cross-validation RMSE, and
            evaluates the selected model on the test dataset.

        Output :
            Returns best model and regression metric artifact.

        On Failure :
            Raises RoundupException.
        """

        try:

            logging.info(
                "Entered get_model_object_and_report method"
            )

            # ---------------------------------------------------------
            # Load model configuration
            # ---------------------------------------------------------

            model_config = read_yaml_file(
                file_path=self.model_trainer_config.model_config_file_path
            )

            randomized_search_config = model_config[
                "randomized_search"
            ]

            model_selection_config = model_config[
                "model_selection"
            ]

            # ---------------------------------------------------------
            # Split train and test data
            # ---------------------------------------------------------

            x_train = train[:, :-1]
            y_train = train[:, -1]

            x_test = test[:, :-1]
            y_test = test[:, -1]

            logging.info(
                f"x_train shape: {x_train.shape}"
            )

            logging.info(
                f"y_train shape: {y_train.shape}"
            )

            logging.info(
                f"x_test shape: {x_test.shape}"
            )

            logging.info(
                f"y_test shape: {y_test.shape}"
            )

            # ---------------------------------------------------------
            # RandomizedSearchCV configuration
            # ---------------------------------------------------------

            randomized_search_class = self._import_class(
                randomized_search_config["module"],
                randomized_search_config["class"]
            )

            randomized_search_params = randomized_search_config[
                "params"
            ]

            # ---------------------------------------------------------
            # Best model variables
            # ---------------------------------------------------------

            best_model = None
            best_model_name = None
            best_cv_rmse = float("inf")

            # ---------------------------------------------------------
            # Train all models
            # ---------------------------------------------------------

            for model_name, model_config_data in model_selection_config.items():

                logging.info(
                    f"Training model: {model_name}"
                )

                # -----------------------------------------------------
                # Import model class
                # -----------------------------------------------------

                model_class = self._import_class(
                    model_config_data["module"],
                    model_config_data["class"]
                )

                # -----------------------------------------------------
                # Model parameters
                # -----------------------------------------------------

                model_params = model_config_data.get(
                    "params",
                    {}
                )

                search_distributions = model_config_data.get(
                    "search_param_distributions",
                    {}
                )

                # -----------------------------------------------------
                # Create model
                # -----------------------------------------------------

                model = model_class(
                    **model_params
                )

                # -----------------------------------------------------
                # Models without hyperparameter search
                # -----------------------------------------------------

                if not search_distributions:

                    logging.info(
                        f"Running cross-validation for {model_name}"
                    )

                    # Use the SAME CV methodology for models
                    # without hyperparameter search.
                    cv_scores = cross_val_score(
                        model,
                        x_train,
                        y_train,
                        cv=randomized_search_params["cv"],
                        scoring="neg_root_mean_squared_error",
                        n_jobs=randomized_search_params["n_jobs"]
                    )

                    cv_rmse = -np.mean(cv_scores)

                    # Fit model on complete training data
                    model.fit(
                        x_train,
                        y_train
                    )

                    current_model = model

                # -----------------------------------------------------
                # Models with RandomizedSearchCV
                # -----------------------------------------------------

                else:

                    logging.info(
                        f"Running RandomizedSearchCV for {model_name}"
                    )

                    randomized_search = randomized_search_class(
                        estimator=model,
                        param_distributions=search_distributions,
                        **randomized_search_params
                    )

                    randomized_search.fit(
                        x_train,
                        y_train
                    )

                    current_model = randomized_search.best_estimator_

                    # RandomizedSearchCV returns negative RMSE
                    cv_rmse = -randomized_search.best_score_

                    logging.info(
                        f"{model_name} best parameters: "
                        f"{randomized_search.best_params_}"
                    )

                # -----------------------------------------------------
                # Log model performance
                # -----------------------------------------------------

                logging.info(
                    f"{model_name} CV RMSE: {cv_rmse}"
                )

                # -----------------------------------------------------
                # Select model with lowest CV RMSE
                # -----------------------------------------------------

                if cv_rmse < best_cv_rmse:

                    best_cv_rmse = cv_rmse
                    best_model = current_model
                    best_model_name = model_name

                    logging.info(
                        f"New best model: {best_model_name}"
                    )

            # ---------------------------------------------------------
            # Check best model
            # ---------------------------------------------------------

            if best_model is None:

                raise Exception(
                    "No suitable regression model was found."
                )

            logging.info(
                f"Best model selected: {best_model_name}"
            )

            logging.info(
                f"Best CV RMSE: {best_cv_rmse}"
            )

            # ---------------------------------------------------------
            # Test prediction
            # ---------------------------------------------------------

            y_pred = best_model.predict(
                x_test
            )

            # ---------------------------------------------------------
            # Calculate regression metrics
            # ---------------------------------------------------------

            mae = mean_absolute_error(
                y_test,
                y_pred
            )

            rmse = root_mean_squared_error(
                y_test,
                y_pred
            )

            r2 = r2_score(
                y_test,
                y_pred
            )

            logging.info(
                f"Test MAE: {mae}"
            )

            logging.info(
                f"Test RMSE: {rmse}"
            )

            logging.info(
                f"Test R2 Score: {r2}"
            )

            # ---------------------------------------------------------
            # Create metric artifact
            # ---------------------------------------------------------

            metric_artifact = RegressionMetricArtifact(
                rmse=rmse,
                mae=mae,
                r2_score=r2
            )

            logging.info(
                f"Regression metric artifact: {metric_artifact}"
            )

            return best_model, metric_artifact

        except Exception as e:

            raise RoundupException(e, sys) from e

    def initiate_model_trainer(
        self,
    ) -> ModelTrainerArtifact:

        """
        Method Name : initiate_model_trainer

        Description :
            Initiates the model training pipeline.

        Output :
            Returns ModelTrainerArtifact.

        On Failure :
            Raises RoundupException.
        """

        logging.info(
            "Entered initiate_model_trainer method"
        )

        try:

            # ---------------------------------------------------------
            # Load transformed train data
            # ---------------------------------------------------------

            train_arr = load_numpy_array_data(
                file_path=(
                    self.data_transformation_artifact
                    .transformed_train_file_path
                )
            )

            # ---------------------------------------------------------
            # Load transformed test data
            # ---------------------------------------------------------

            test_arr = load_numpy_array_data(
                file_path=(
                    self.data_transformation_artifact
                    .transformed_test_file_path
                )
            )

            logging.info(
                "Loaded transformed train and test arrays"
            )

            # ---------------------------------------------------------
            # Train models
            # ---------------------------------------------------------

            best_model, metric_artifact = (
                self.get_model_object_and_report(
                    train=train_arr,
                    test=test_arr
                )
            )

            # ---------------------------------------------------------
            # Load preprocessing object
            # ---------------------------------------------------------

            preprocessing_obj = load_object(
                file_path=(
                    self.data_transformation_artifact
                    .transformed_object_file_path
                )
            )

            # ---------------------------------------------------------
            # Create RoundupModel
            # ---------------------------------------------------------

            roundup_model = RoundupModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=best_model
            )

            logging.info(
                "Created RoundupModel object"
            )

            # ---------------------------------------------------------
            # Save trained model
            # ---------------------------------------------------------

            save_object(
                self.model_trainer_config.trained_model_file_path,
                roundup_model
            )

            logging.info(
                "Saved trained model successfully"
            )

            # ---------------------------------------------------------
            # Create ModelTrainerArtifact
            # ---------------------------------------------------------

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=(
                    self.model_trainer_config.trained_model_file_path
                ),
                metric_artifact=metric_artifact
            )

            logging.info(
                f"Model trainer artifact: "
                f"{model_trainer_artifact}"
            )

            return model_trainer_artifact

        except Exception as e:

            raise RoundupException(e, sys) from e