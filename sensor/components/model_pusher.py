import os, sys
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.utils import load_object, save_object
from sensor.entity.config_entity import ModelPusherConfig
from sensor.entity.artifact_entity import DataTransformationArtifact, ModelPusherArtifact, ModelTrainerArtifact
from sensor.predictor import ModelResolver

class ModelPusher:

    def __init__(self, model_pusher_config:ModelPusherConfig,
                 data_transformation_artifact:DataTransformationArtifact,
                 model_trainer_artifact:ModelTrainerArtifact):
        try:
            logging.info(f"{'>>'*20} Model Pusher {'<<'*20}")
            self.model_pusher_config = model_pusher_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver(model_registry=self.model_pusher_config.saved_model_dir)

        except Exception as e:
            raise SensorException(e, sys)
        

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:

            # Load Objects
            logging.info("Loading Transformer, Model and Target Encoder objects")
            transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)
            model = load_object(file_path=self.model_trainer_artifact.model_path)
            target_encoder = load_object(file_path=self.data_transformation_artifact.target_encoder_path)

            # Saving objects in Model Pusher Directory
            logging.info("Saving objects into Model Pusher Directory")
            save_object(file_path=self.model_pusher_config.pusher_transformer_path, obj=transformer)
            save_object(file_path=self.model_pusher_config.pusher_model_path, obj=model)
            save_object(file_path=self.model_pusher_config.pusher_target_encoder_path, obj=target_encoder)

            # Saving Objects in Saved Model Directory
            logging.info(f"Saving objects in Saved Model Directory")
            saved_transformer_path = self.model_resolver.get_latest_save_transformer_path()
            saved_model_path = self.model_resolver.get_latest_save_model_path()
            saved_target_encoder_path = self.model_resolver.get_latest_save_target_encoder_path()

            save_object(file_path=saved_transformer_path, obj=transformer)
            save_object(file_path=saved_model_path, obj=model)
            save_object(file_path=saved_target_encoder_path, obj=target_encoder)

            model_pusher_artifact = ModelPusherArtifact(pusher_model_dir=self.model_pusher_config.pusher_model_dir,
                                                        saved_model_dir=self.model_pusher_config.saved_model_dir)
            
            logging.info(f"Model Pusher Artifact: {model_pusher_artifact}")

            return model_pusher_artifact
        
        except Exception as e:
            raise SensorException(e, sys)