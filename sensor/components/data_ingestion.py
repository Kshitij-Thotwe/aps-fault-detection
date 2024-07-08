import os, sys
import pandas as pd
import numpy as np
from sensor import utils
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity import config_entity
from sensor.entity import artifact_entity
from sklearn.model_selection import train_test_split


class DataIngestion:

    def __init__(self, data_ingestion_config:config_entity.DataIngestionConfig):
        
        try:
            logging.info(f"{'>>'*20} Data Ingestion {'<<'*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise SensorException(e, sys)
        
    
    def initiate_data_ingestion(self) -> artifact_entity.DataIngestionArtifact:
        
        try:

            logging.info(f"Exporting collection data as Pandas Dataframe")
            # Exporting collection data in a pandas dataframe
            df:pd.DataFrame = utils.get_collection_as_dataframe(database_name=self.data_ingestion_config.database_name,
                                                                collection_name=self.data_ingestion_config.collection_name)
            
            
            # Replace na values with NAN
            df.replace(to_replace="na", value=np.NAN, inplace=True)

            
            logging.info(f"Saving data in feature store")
            # making a feature store folder
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            logging.info(f"Make new folder store folder if not available")
            # make new folder if not available
            os.makedirs(feature_store_dir, exist_ok=True)
            logging.info(f"Save df to feature folder")
            # Store the df in feature store folder
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_file_path, index=False, header=True)

            
            logging.info(f"Split the dataset into train and test set")
            # splitting the df into train and test
            train_df, test_df = train_test_split(df, test_size=self.data_ingestion_config.test_size, random_state=40)

            
            logging.info(f"Create Dataset directory if not available")
            # making dataset folder
            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            # make new folder if not available
            os.makedirs(dataset_dir, exist_ok=True)

            
            logging.info(f"Save train and test set in dataset folder")
            # Store in dataaset folder
            train_df.to_csv(path_or_buf=self.data_ingestion_config.train_file_path, index=False, header=True)
            test_df.to_csv(path_or_buf=self.data_ingestion_config.test_file_path, index=False, header=True)

            
            # Preparing Artifacts

            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path)
            
            logging.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        
        except Exception as e:
            raise SensorException(e, sys)