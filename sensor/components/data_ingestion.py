import pandas as pd
import numpy as np
import os, sys
from sensor import utils
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.entity import config_entity
from sensor.entity import artifact_entity
from sklearn.model_selection import train_test_split

class DataIngestion:

    def __init__(self, data_ingestion_config:config_entity.DataIngestionConfig):
        
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise SensorException(e, sys)
        
    def initiate_data_ingestion(self) -> artifact_entity.DataIngestionArtifact:

        try:

            # Export data to pandas dataframe
            df:pd.DataFrame = utils.get_collection_as_dataframe(database_name=self.data_ingestion_config.database_name,
                                                              collection_name=self.data_ingestion_config.collection_name)
            
            # replace na values
            df.replace(to_replace="na", value=np.NAN, inplace=True)

            # feature folder
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_path)
            # make new if not available
            os.makedirs(feature_store_dir, exist_ok=True)

            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_path, index=False, header=True)

            # split dataset
            train_df,test_df = train_test_split(df,test_size=self.data_ingestion_config.test_size,random_state=42)

            # dataset directory
            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)

            # make new if not available
            os.makedirs(dataset_dir, exist_ok=True)

            train_df.to_csv(path_or_buf=self.data_ingestion_config.train_file_path,index=False,header=True)
            test_df.to_csv(path_or_buf=self.data_ingestion_config.test_file_path, index=False, header=True)

            # Preparing Artifact

            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_path,
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path)
            

            return data_ingestion_artifact
        
        except Exception as e:
            raise SensorException(e, sys)
                      
            
