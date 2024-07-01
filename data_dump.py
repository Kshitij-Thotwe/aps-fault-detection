import pymongo
import pandas as pd
import json


# Provide the MongoDB localhost url to connect python to mongodb
client = pymongo.MongoClient("mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.2.10")

DATA_FILE_PATH="C:/Users/Dell/Desktop/Projects/VS Code/aps_failure_training_set1.csv"
DATABASE_NAME="aps"
COLLECTION_NAME="sensor"

if __name__=="__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Rows and Columns: {df.shape}")

    # Converting DataFrame to JSON to dump these records in mongodb
    df.reset_index(drop=True, inplace=True)

    json_record = list(json.loads(df.T.to_json()).values())

    # insert converted json record to mongodb
    client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)