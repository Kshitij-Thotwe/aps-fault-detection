import pandas as pd
import numpy as np
import os, sys
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity import config_entity, artifact_entity
from sensor import utils
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

class ModelTrainer:

    def __init__(self, model_trainer_config:config_entity.ModelTrainerConfig,
                       data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise SensorException(e, sys)
        

    def train_model(self, x, y):
        try:
            xgb_clf = XGBClassifier()
            xgb_clf.fit(x, y)
            return xgb_clf
        
        except Exception as e:
            raise SensorException(e, sys)
        

    def initiate_model_trainer(self) -> artifact_entity.ModelTrainerArtifact:
        try:

            logging.info(f"Loading Train and Test array")
            train_df = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_df = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)

            logging.info(f"Splitting input and target feature from both Train and Test array")
            x_train, y_train = train_df[:, :-1], train_df[:, -1]
            x_test, y_test = test_df[:, :-1], test_df[:, -1]

            logging.info(f"Train Model")
            model = self.train_model(x_train, y_train)

            logging.info(f"Calculating f1 train score")
            ycap_train = model.predict(x_train)
            f1_train_score = f1_score(y_true=y_train, y_pred=ycap_train)

            logging.info(f"Calculating f1 test score")
            ycap_test = model.predict(x_test)
            f1_test_score = f1_score(y_true=y_test, y_pred=ycap_test)

            logging.info(f"Train score: {f1_train_score}  and  Test Score: {f1_test_score}")
            
            logging.info(f"Checking if our model is underfitting or not")
            
            if f1_test_score < self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give \
                                 expected accuracy: {self.model_trainer_config.expected_score}, model actual score: {f1_test_score}")
            
            logging.info(f"Checking if our model is overfitting or not")
            diff = abs(f1_test_score-f1_train_score)

            if diff > self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and Test diff: {diff} is more than the overfitting threshold: {self.model_trainer_config.overfitting_threshold}")

            logging.info(f"Saving Model Object")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)

            logging.info(f"Preparing Artifact")
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(
                model_path=self.model_trainer_config.model_path,
                f1_train_score=f1_train_score,
                f1_test_sccore=f1_test_score
            )

            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise SensorException(e, sys)