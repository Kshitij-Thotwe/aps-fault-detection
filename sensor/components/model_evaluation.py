import pandas as pd
import os, sys
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.config import TARGET_COLUMN
from sensor.entity import config_entity, artifact_entity
from sensor.predictor import ModelResolver
from sensor.utils import load_object
from sklearn.metrics import f1_score

class ModelEvaluation:

    def __init__(self,
                 model_eval_config:config_entity.ModelEvaluationConfig,
                 data_transformation_artifact:artifact_entity.DataTransformationArtifact,
                 data_ingestion_artifact:artifact_entity.DataIngestionArtifact,
                 model_trainer_artifact:artifact_entity.ModelTrainerArtifact):
        try:
            logging.info(f"{'>>'*20} Model Evalutaion {'<<'*20}")
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver()

        except Exception as e:
            raise SensorException(e, sys)
        

    def initiate_model_evalutaion(self) -> artifact_entity.ModelEvaluationArtifact:
        try:
            logging.info("If saved model folder already has a model, then we will compare which model is best trained: "
                         "The saved model or the current model")

            latest_dir_path = self.model_resolver.get_latest_dir_path()

            if latest_dir_path == None:
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
                                                                              improved_accuracy=None)
                logging.info(f"Model Evaluation Artifact: {model_eval_artifact}")
                
                return model_eval_artifact
            
            logging.info("Finding locations of saved transformer, model and target encoder objects")
            transformer_path = self.model_resolver.get_latest_transformer_path()
            model_path = self.model_resolver.get_latest_model_path()
            target_encoder_path = self.model_resolver.get_latest_target_encoder_path()

            logging.info("Previously Trained Objects: Transformer, Model and Target Encoder")
            transformer = load_object(file_path=transformer_path)
            model = load_object(file_path=model_path)
            target_encoder = load_object(file_path=target_encoder_path)

            
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            target_df = test_df[TARGET_COLUMN]
            y_true = target_encoder.transform(target_df)

            # Accuracy using Previous Model
            input_feature = list(transformer.feature_names_in_)
            input_arr = transformer.transform(test_df[input_feature])
            y_pred = model.predict(input_arr)
            print(f"Prediction using previous model: {target_encoder.inverse_transform(y_pred[:5])}")

            previous_model_accuracy = f1_score(y_true=y_true, y_pred=y_pred)
            logging.info(f"Accuracy using Previous Model: {previous_model_accuracy}")

            logging.info("Currently trained model objects")
            # Currently trained model objects
            current_transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)
            current_model = load_object(file_path=self.model_trainer_artifact.model_path)
            current_target_encoder = load_object(file_path=self.data_transformation_artifact.target_encoder_path)

            # Accuracy using Current Model
            input_feature_name = list(current_transformer.feature_names_in_)
            input_arr_current = current_transformer.transform(test_df[input_feature_name])
            y_pred_current = current_model.predict(input_arr_current)
            current_target_encoder.inverse_transform(y_pred_current[:5])
            print(f"Prediction using previous model: {current_target_encoder.inverse_transform(y_pred_current[:5])}")

            current_model_accuracy = f1_score(y_true=y_true, y_pred=y_pred_current)
            logging.info(f"Accuracy using Current Model: {current_model_accuracy}")

            if current_model_accuracy < previous_model_accuracy:
                logging.info("Current Trained Model does not perform better than Previous Trained Model")
                raise Exception("Current Trained Model does not perform better than Previous Trained Model")
            
            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
                                                                          improved_accuracy=current_model_accuracy-previous_model_accuracy)
            
            logging.info(f"Model Evaluation Artifact: {model_eval_artifact}")
            
            return model_eval_artifact
        
        except Exception as e:
            raise SensorException(e, sys)