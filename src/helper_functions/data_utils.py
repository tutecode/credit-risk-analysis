from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder,RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, chi2

# import sys
# sys.path.append('..')
import os
from typing import Tuple
import os
import boto3
# from pathlib import Path
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

def get_feature_in_set(
    app_train: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Separates our train and test datasets columns between Features
    (the input to the model) and Targets (what the model has to predict with the
    given features).

    Arguments:
        app_train : pd.DataFrame
            Training datasets

    Returns:
        app_train : pd.DataFrame
            Training features
        app_val : pd.DataFrame
            Validation features
        app_test : pd.DataFrame
            Testing features
    """
    app_train_set, app_val_set, app_test_set = None, None, None
    
    app_temp_set, app_test_set = train_test_split(app_train, test_size=0.2,random_state=40)
    app_train_set, app_val_set = train_test_split(app_temp_set, test_size=0.1,random_state=0)

    return app_train_set, app_val_set, app_test_set

def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # Taking the columns that contain objects.
    category_columns = working_train_df.select_dtypes(exclude="number").columns.to_list()
    print("cat_cols: ", working_train_df.select_dtypes(exclude="number").columns)
    numeric_columns = working_train_df.select_dtypes(include="number").columns.to_list()
    print(numeric_columns)
    # Filtering the dataset.
    aux_dataframe = working_train_df[category_columns].copy()
    mask_2 = (aux_dataframe.nunique() == 2).values
    cat_2 = aux_dataframe.loc[:, mask_2].columns
    print(cat_2)
    mask_gt_2 = (aux_dataframe.nunique() > 2).values
    cat_gt_2 = aux_dataframe.loc[:, mask_gt_2].columns
    print(cat_gt_2)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy='median')), 
            ("scaler", RobustScaler())
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy='most_frequent')),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ("scaler", RobustScaler())
        ]
    )
    bincategorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy='most_frequent')),
            ("encoder", OrdinalEncoder()),
        ]
    )
    
    ct_preprocessing = ColumnTransformer(transformers=[
        ('transform_num', numeric_transformer, numeric_columns),
        ('transform_cat', categorical_transformer, cat_gt_2),
        ('transform_bin', bincategorical_transformer, cat_2),
        
    ], remainder='passthrough')

    ct_preprocessing.fit(working_train_df)
    # # imputer.set_output(transform="pandas")

    working_train_df = ct_preprocessing.transform(working_train_df)
    working_val_df = ct_preprocessing.transform(working_val_df)
    working_test_df = ct_preprocessing.transform(working_test_df)

    return working_train_df, working_val_df, working_test_df

def get_feature_target(
    app_train: pd.DataFrame, app_val: pd.DataFrame, app_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series,pd.DataFrame, pd.Series]:
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
    X_train, y_train, X_val, y_val,X_test, y_test = None, None, None, None,None,None

    # training
    X_train = app_train[:,:-1]
    y_train = app_train[:,-1:]

    # validation
    X_val = app_val[:,:-1]
    y_val = app_val[:,-1:]

    
    # testing
    X_test = app_test[:,:-1]
    y_test = app_test[:,-1:]

    return X_train, y_train, X_val, y_val, X_test,y_test