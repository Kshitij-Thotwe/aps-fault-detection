import pandas as pd
import numpy as np
import os, sys
from typing import Optional
from sensor.entity import config_entity, artifact_entity
from sensor.logger import logging
from sensor.exception import SensorException
from sensor import utils
from sensor.config import TARGET_COLUMN
from scipy.stats import ks_2samp


class DataValidation:

    def __init__(self, data_validation_config:config_entity.DataValidationConfig,
                       data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data Validation {'<<'*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.validation_error = dict()
        except Exception as e:
            raise SensorException(e, sys)
        

    def dropped_missing_column_values(self, df:pd.DataFrame, report_key_name:str) -> Optional[pd.DataFrame]:
        """
        This function will drop column which contains missing value more than specified threshold

        df: Accepts a pandas dataframe
        threshold: Percentage criteria to drop a column
        =====================================================================================
        returns Pandas DataFrame if atleast a single column is available after missing columns drop else None
        """

        try:

            threshold = self.data_validation_config.threshold
            null_report = df.isnull().sum() / df.shape[0]

            logging.info(f"Selecting Column names which contain null values above {threshold}")
            drop_column_names = null_report[null_report>threshold].index

            logging.info(f"Columns to drop: {list(drop_column_names)}")
            self.validation_error[report_key_name] = list(drop_column_names)
            df.drop(list(drop_column_names), axis=1, inplace=True)

            if len(df.columns)==0:
                return None
            return df
        
        except Exception as e:
            raise SensorException(e, sys)
        

    def do_required_columns_exists(self, base_df:pd.DataFrame, current_df:pd.DataFrame, report_key_name:str)->bool:
        try:

            base_columns = base_df.columns
            current_columns = current_df.columns

            missing_columns = []

            for base_column in base_columns:
                if base_column not in current_columns:
                    logging.info(f"Column: [{base_column} is not available]")
                    missing_columns.append(base_column)

            if len(missing_columns)>0:
                self.validation_error[report_key_name] = missing_columns
                return False
            return True
        
        except Exception as e:
            raise SensorException(e, sys)
        

    def data_drift(self, base_df:pd.DataFrame, current_df:pd.DataFrame, report_key_name:str):
        try:
            drift_report = dict()

            base_columns = base_df.columns
            current_columns = current_df.columns

            for base_column in base_columns:
                base_data, current_data = base_df[base_column], current_df[base_column]

                logging.info(f"Hypothesis {base_column}: {base_data.dtype}, {current_data.dtype} ")
                same_distribution = ks_2samp(base_data, current_data)

                if same_distribution.pvalue>0.05:
                    drift_report[base_column] = {
                        "pvalues" : float(same_distribution.pvalue),
                        "same_distribution" : True
                    }

                else:
                    drift_report[base_column] = {
                        "pvalues": float(same_distribution.pvalue),
                        "same_distribution" : False
                    }

            self.validation_error[report_key_name] = drift_report

        except Exception as e:
            raise SensorException(e, sys)
        

    def initiate_data_validation(self) -> artifact_entity.DataValidationArtifact:
        try:

            logging.info(f"Reading base DataFrame")
            base_df = pd.read_csv(self.data_validation_config.base_file_path)

            logging.info(f"Replace na values in base_df")
            base_df.replace(to_replace="na", value=np.NAN, inplace=True)

            logging.info(f"Drop Null value columns from base_df above threshold")
            base_df = self.dropped_missing_column_values(df=base_df, report_key_name="missing_values_within_base_dataset")

            logging.info(f"Reading Train and Test DataFrame")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info(f"Dropping Null Value columns from train_df and test_df")
            train_df = self.dropped_missing_column_values(df=train_df, report_key_name="missing_values_within_train_dataset")
            test_df = self.dropped_missing_column_values(df=test_df, report_key_name="missing_values_within_test_dataset")

            exclude_columns = [TARGET_COLUMN]

            base_df = utils.convert_column_float(df=base_df, exclude_columns=exclude_columns)
            train_df = utils.convert_column_float(df=train_df, exclude_columns=exclude_columns)
            test_df = utils.convert_column_float(df=test_df, exclude_columns=exclude_columns)

            logging.info(f"Are all required columns present in train_df and test_df")
            train_df_column_status = self.do_required_columns_exists(base_df=base_df, current_df=train_df, report_key_name="missing_columns_within_train_dataset")
            test_df_column_status = self.do_required_columns_exists(base_df=base_df, current_df=test_df, report_key_name="missing_columns_within_test_dataset")

            if train_df_column_status:
                logging.info(f"As all columns are available in train_df, hence detecting data_drift")
                self.data_drift(base_df=base_df, current_df=train_df, report_key_name="data_drift_within_train_data")

            if test_df_column_status:
                logging.info(f"As all columns are available in test_df, hence detecting data_drift")
                self.data_drift(base_df=base_df, current_df=test_df, report_key_name="data_drift_within_test_dataset")

            logging.info(f"Write report in yaml file")
            utils.write_yaml_file(file_path=self.data_validation_config.report_file_path, data=self.validation_error)

            data_validation_artifact = artifact_entity.DataValidationArtifact(report_file_path=self.data_validation_config.report_file_path)
            logging.info(f"Data Validation Artifact: {data_validation_artifact}")

            return data_validation_artifact
        
        except Exception as e:
            raise SensorException(e, sys)





