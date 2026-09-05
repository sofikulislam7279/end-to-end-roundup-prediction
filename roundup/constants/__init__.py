import os
from datetime import date


# Database Configuration
DATABASE_NAME = "roundupdb"
COLLECTION_NAME = "roundup_data"
MONGODB_URL_KEY = "MONGODB_URL"


# Pipeline Configuration
PIPELINE_NAME: str = "roundup_regression"


# Artifact Configuration
ARTIFACT_DIR: str = "artifact"
MODEL_FILE_NAME = "model.pkl"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"


# Dataset Configuration
FILE_NAME: str = "roundup.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
TARGET_COLUMN = "roundup_amount"


# Schema Configuration
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")


# General Configuration
CURRENT_YEAR = date.today().year


# Data Ingestion Configuration
DATA_INGESTION_COLLECTION_NAME: str = "roundup_data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2


# Data Validation Configuration
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

# Data Transformation Configuration
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
