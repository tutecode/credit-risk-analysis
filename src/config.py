import os
from pathlib import Path

BUCKET = 'anyoneai-datasets'
PREFIX = "credit-data-2010"

DATASET_ROOT_PATH = str(Path(__file__).parent.parent / "dataset")
os.makedirs(DATASET_ROOT_PATH, exist_ok=True)

DATASET_CREDIT = str(Path(DATASET_ROOT_PATH) / "credit_data.csv")
DATASET_CREDIT_URL = f'{PREFIX}/PAKDD2010_Modeling_Data.txt'

# DATASET_TRAIN = str(Path(DATASET_ROOT_PATH) / "train_data.csv")
# DATASET_TRAIN_URL = f'{PREFIX}/PAKDD2010_Modeling_Data.txt'

# DATASET_TEST = str(Path(DATASET_ROOT_PATH) / "test_data.csv")
# DATASET_TEST_URL =  f'{PREFIX}/PAKDD2010_Prediction_Data.txt'

DATASET_DESCRIPTION= str(Path(DATASET_ROOT_PATH) / "description.xls")
DATASET_DESCRIPTION_URL = f'{PREFIX}/PAKDD2010_VariablesList.XLS'

# DATASET_LEADERBOARD = str(Path(DATASET_ROOT_PATH) / "leaderboard_data.csv")
# DATASET_TEADERBOARD_URL = f'{PREFIX}/PAKDD2010_Leaderboard_Submission_Example.txt'
