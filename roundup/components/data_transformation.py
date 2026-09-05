import sys

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from roundup.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from roundup.entity.config_entity import DataTransformationConfig
from roundup.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from roundup.exception import RoundupException
from roundup.logger import logging

from roundup.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):

        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """

        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)

        except Exception as e:
            raise RoundupException(e, sys)


    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise RoundupException(e, sys) from e


    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Method Name :   get_data_transformer_object
        Description :   This method creates and returns a data transformer object for the data
        
        Output      :   data transformer object is created and returned 
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            logging.info("Got columns from schema config to transform")

            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder(handle_unknown="ignore")

            logging.info("Initialized StandardScaler, OneHotEncoder")

            num_features = self._schema_config['num_features']
            oh_columns = self._schema_config['oh_columns']

            logging.info("Initialize ColumnTransformer")

            preprocessor = ColumnTransformer(
                [
                    ("StandardScaler", numeric_transformer, num_features),
                    ("OneHotEncoder", oh_transformer, oh_columns)
                ]
            )

            logging.info("Created preprocessor object from ColumnTransformer")

            logging.info("Exited get_data_transformer_object method of DataTransformation class")

            return preprocessor

        except Exception as e:
            raise RoundupException(e, sys) from e


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Method Name :   initiate_data_transformation
        Description :   This method initiates the data transformation component for the pipeline 
        
        Output      :   data transformer steps are performed and preprocessor object is created  
        On Failure  :   Write an exception log and then raise an exception
        """

        try:
            if self.data_validation_artifact.validation_status:
                logging.info("Starting data transformation")
                preprocessor = self.get_data_transformer_object()
                logging.info("Got the preprocessor object")

                train_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
                test_df = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

                # Train 
                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN]

                logging.info("Loaded training dataset")

                drop_cols = self._schema_config['drop_columns']

                logging.info("Dropping configured columns from training dataset")

                input_feature_train_df = drop_columns(df=input_feature_train_df, cols = drop_cols)

                # Test 
                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_test_df = test_df[TARGET_COLUMN]

                input_feature_test_df = drop_columns(df=input_feature_test_df, cols = drop_cols)

                logging.info("Dropping configured columns from test dataset")
                logging.info("Loaded testing dataset")

                logging.info("Applying preprocessing object on training dataframe and testing dataframe")

                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)

                logging.info("Used the preprocessor object to fit transform the train features")

                input_feature_test_arr = preprocessor.transform(input_feature_test_df)

                logging.info("Used the preprocessor object to transform the test features")

                logging.info("Created train array and test array")

                train_arr = np.c_[
                    input_feature_train_arr, np.array(target_feature_train_df)
                ]

                test_arr = np.c_[
                    input_feature_test_arr, np.array(target_feature_test_df)
                ]

                save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
                save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

                logging.info("Saved the preprocessor object")

                logging.info("Exited initiate_data_transformation method of Data_Transformation class")

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )

                return data_transformation_artifact

            else:
                raise Exception(self.data_validation_artifact.message)


        except Exception as e:
            raise RoundupException(e, sys) from e


        





