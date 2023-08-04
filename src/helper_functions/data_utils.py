from sklearn.model_selection import train_test_split
from imblearn import under_sampling


# import sys
# sys.path.append('..')
import os
from typing import Tuple
import os
import boto3
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from helper_functions import config
import numpy as np

def get_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Download from S3 all the needed datasets for the project.

    Returns:
        app_train : pd.DataFrame
            Training dataset

        app_test : pd.DataFrame
            Test dataset

        description : pd.DataFrame
            Extra dataframe with detailed description about dataset features
    """
    load_dotenv()

    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )
    
    # Download application_train.csv
    if not os.path.exists(config.DATASET_TRAIN):
        s3.download_file(config.BUCKET, config.DATASET_TRAIN_URL, f'{config.DATASET_ROOT_PATH}/train_data.csv')
    
    # Download application_test.csv
    if not os.path.exists(config.DATASET_TEST):
        s3.download_file(config.BUCKET, config.DATASET_TEST_URL, f'{config.DATASET_ROOT_PATH}/test_data.csv')

    # Download description.xls
    if not os.path.exists(config.DATASET_DESCRIPTION):
        s3.download_file(config.BUCKET, config.DATASET_DESCRIPTION_URL, f'{config.DATASET_ROOT_PATH}/description.xls')

    # create datasets
    df_excel = pd.read_excel(config.DATASET_DESCRIPTION, index_col=0, engine="xlrd")
    app_train = pd.read_csv(config.DATASET_TRAIN, delimiter='\t', encoding='latin1', header=None)
    app_test = pd.read_csv(config.DATASET_TEST, delimiter='\t', encoding='latin1', header=None)

    app_train.columns = df_excel['Var_Title'].to_list()
    app_test.columns = df_excel['Var_Title'].to_list()[:-1]

    return (app_train, app_test, df_excel)


# to create a csv file from a pandas df
def df_csv(df, name):
    new_path = str(Path(config.DATASET_ROOT_PATH) / name)
    df.to_csv(new_path, index=False)
    print(f'The file has been saved in: {new_path}')


# to load a specific csv as a pandas df
def get_normalized_model():
    app_normalized = pd.read_csv(config.DATASET_NORMALIZED, encoding='latin1')
    return app_normalized


# train, test, val split
def get_feature(df, target_col = 'TARGET_LABEL_BAD=1'):
    
    target = df[target_col]
    df_train = df.drop(columns=[target_col])
    X_temp, X_test, y_temp, y_test = train_test_split(df_train, target, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, random_state=42)

    return X_train, y_train, X_test, y_test, X_val, y_val


# resampling
def resampling(X, y, sampling_strategy = 1.0):

    rus = under_sampling.RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    X_resampling, y_resampling = rus.fit_resample(X, y)
    return X_resampling, y_resampling

