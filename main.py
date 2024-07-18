import os, sys
from sensor.exception import SensorException
from sensor.pipeline.training_pipeline import start_training_pipeline
from sensor.pipeline.batch_prediction import start_batch_prediction

input_dir = "aps_failure_training_set1.csv"
print(__name__)


if __name__ == "__main__":
    try:
        start_training_pipeline()
        output = start_batch_prediction(input_file_path=input_dir)
        print(f"Prediction Complete, file stored here: {output}")

    except Exception as e:
        raise SensorException(e, sys)