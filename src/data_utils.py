from sklearn.model_selection import train_test_split
import os
from typing import Tuple
import os
import boto3
# from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from src import config

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


def get_feature_target(
    app_train: pd.DataFrame, app_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Separates our train and test datasets columns between Features
    (the input to the model) and Targets (what the model has to predict with the
    given features).

    Arguments:
        app_train : pd.DataFrame
            Training datasets
        app_test : pd.DataFrame
            Test datasets

    Returns:
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test target
    """
    X_train, y_train, X_test, y_test = None, None, None, None

    X = app_train.drop(columns=['TARGET_LABEL_BAD=1'])
    y = app_train['TARGET_LABEL_BAD=1']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y,random_state=0)
    X_test = app_test

    return X_train, y_train, X_val, y_val, X_test
