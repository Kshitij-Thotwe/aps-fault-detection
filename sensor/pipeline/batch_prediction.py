import numpy as np
import pandas as pd
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.utils import load_object
from sensor.predictor import ModelResolver
import os, sys
from datetime import datetime
PREDICTION_DIR = "prediction"

def start_batch_prediction(input_file_path):
    try:

        os.makedirs(PREDICTION_DIR, exist_ok=True)

        logging.info("Creating Model Resolver Object")
        model_resolver = ModelResolver(model_registry="saved_models")

        logging.info(f"Reading file: {input_file_path}")
        df = pd.read_csv(input_file_path)
        df.replace(to_replace="na", value=np.NAN, inplace=True)

        logging.info("Loading Transformer to transform loaded dataset")
        transformer = load_object(file_path=model_resolver.get_latest_transformer_path())

        input_feature_name = list(transformer.feature_names_in_)
        input_arr = transformer.transform(df[input_feature_name])

        logging.info("Loading Model to make prediction")
        model = load_object(file_path=model_resolver.get_latest_model_path())
        prediction = model.predict(input_arr)

        logging.info("Loading Target Encoder to convert predicted column to categorical values")
        target_encoder = load_object(file_path=model_resolver.get_latest_target_encoder_path())
        cat_pred = target_encoder.inverse_transform(prediction)

        df["prediction"] = prediction
        df["cat_pred"] = cat_pred

        prediciton_file_name = os.path.basename(input_file_path).replace(".csv", f"{datetime.now().strftime('%m%d%H__%H%M%S')}.csv")
        prediciton_file_path = os.path.join(PREDICTION_DIR, prediciton_file_name)

        df.to_csv(prediciton_file_path, index=False, header=True)

        logging.info(f"Batch Prediction Complete, file stored here: {prediciton_file_path}")
        return prediciton_file_path
    
    except Exception as e:
        raise SensorException(e, sys)