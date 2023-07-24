from tabulate import tabulate
import pandas as pd

def display_missing_values(data_frame):
    """
    Displays the columns with missing values and their corresponding counts.
    Args:
        data_frame (pd.DataFrame): The DataFrame to analyze.
    """
    missing_counts = data_frame.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    missing_data = pd.DataFrame(missing_counts, columns=["Missing Count"])
    missing_data.index.name = "Column"

    print(tabulate(missing_data, headers="keys", tablefmt="pretty"))

def display_metadata(data_df,data_cols,metadata_df,metadata_col2idx):
    raw_data = [(number_field,len(data_df[number_field].unique()),metadata_df.iloc[metadata_col2idx[number_field], 2])for number_field in data_cols]
    data_tabulate = pd.DataFrame(raw_data,columns=["Numeric Features","UniqueCount","RangeMeta"])
    print(tabulate(data_tabulate, headers="keys", tablefmt="pretty"))
    