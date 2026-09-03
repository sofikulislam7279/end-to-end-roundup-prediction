from roundup.configuration.mongodb_connection import MongoDBClient
from roundup.constants import DATABASE_NAME
from roundup.exception import RoundupException
import sys
import pandas as pd
import numpy as np
from typing import Optional


class RoundupData:
    """
    This class help to export entire mongodb record as dataframe
    """

    def __init__(self):
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise RoundupException(e, sys)


    def export_collection_as_dataframe(self, collection_name: str, database_name: Optional[str]=None)->pd.DataFrame:
        """
        export entire collection as dataframe
        return pd.DataFrame of collection
        """
        try:
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"])

            df.replace({"na":np.nan}, inplace=True)

            return df
        except Exception as e:
            raise RoundupException(e, sys)


        