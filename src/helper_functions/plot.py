import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tabulate import tabulate

def compute_stats_count(train, field, counting=False):
    count, index, perc = [], [], []
    if counting:
        count = train[field].value_counts().values
        index = train[field].value_counts().index
        perc = train[field].value_counts().values / len(train[field]) * 100
    else:
        count = train[field].values
        index = train[field].index
        perc = train[field].values / train[field].sum() * 100

    return list(zip(index, count, perc))


def percented_patches_first(ax, col_stats):
    patches = ax.patches
    for i in range(len(patches)):
        x = patches[i].get_y() + patches[i].get_height() / 2
        y = patches[i].get_width() + 0.05
        ax.annotate("{:.2f}%".format(col_stats[i][2]), (y, x), va="center")


def percented_patches_second(ax, col_stats):
    patches = ax.patches
    for i in range(len(patches)):
        x = patches[i].get_y() + patches[i].get_height() / 2
        y = patches[i].get_width() + 0.05
        ax.annotate("{:.2f}%".format(col_stats[i]), (y, x), va="center")


def plot_data_type_counts(data_frame):
    """
    Plots the number of columns per data type in the DataFrame.
    Args:
        data_frame (pd.DataFrame): The DataFrame to analyze.
    """
    float_fields = len(data_frame.select_dtypes("float64").columns)
    int_fields = len(data_frame.select_dtypes("int64").columns)
    object_fields = len(data_frame.select_dtypes("object").columns)

    d = {
        "var_type": ["float64", "int64", "object"],
        "quantity": [float_fields, int_fields, object_fields],
    }
    quant_kind_vars = pd.DataFrame(data=d, index=[1, 2, 3])
    plt.figure(figsize=(6, 2))
    ax = sns.barplot(y=quant_kind_vars["var_type"], x=quant_kind_vars["quantity"])
    ax.set_title("Number of Features")
    ax.set_ylabel("Type of Variable")
    target_dist = compute_stats_count(quant_kind_vars, "quantity")
    percented_patches_first(ax, target_dist)
    plt.show()


def plot_target_variable_distribution(data_frame, target_colname):
    """
    Plots the distribution of the target variable in the DataFrame.
    Args:
        data_frame (pd.DataFrame): The DataFrame containing the target variable.
        target_colname (str): The name of the target variable column.
    """
    target_dist = compute_stats_count(data_frame, target_colname, counting=True)

    # Convert the target_dist list to a DataFrame
    target_df = pd.DataFrame(target_dist, columns=["Value", "Count", "Percentage"])

    # Print the target distribution
    print(target_df)

    # Plot the countplot
    plt.figure(figsize=(6, 2))
    plt.title("Target Variable Distribution")
    ax = sns.countplot(data=data_frame, y=target_colname)
    percented_patches_second(
        ax, target_df["Percentage"].values
    )  # Pass the values as a list
    plt.tight_layout()
    plt.show()


def plot_unique_value_counts(data_frame, top_n=10):
    """
    Plots the count of unique values for each categorical column in the DataFrame.
    Args:
        data_frame (pd.DataFrame): The DataFrame to analyze.
        top_n (int): Number of top categorical columns to display (default is 10).
    """
    object_field_name = data_frame.select_dtypes("object").columns.to_list()

    name_features = [object_field for object_field in object_field_name]
    count_unique = [
        len(data_frame[object_field].unique()) for object_field in object_field_name
    ]
    rep_count_unique = sorted(
        zip(name_features, count_unique), key=lambda x: x[1], reverse=True
    )[:top_n]

    rep_count_unique_df = pd.DataFrame(
        rep_count_unique, columns=["NameFeature", "CountUnique"]
    )

    plt.figure(figsize=(6, 2))
    ax = sns.barplot(
        y=rep_count_unique_df["NameFeature"], x=rep_count_unique_df["CountUnique"]
    )
    ax.set_title("Count of Uniques")
    ax.set_ylabel("Feature Name")
    target_dist = compute_stats_count(rep_count_unique_df, "CountUnique")
    percents = [i[2] for i in target_dist]
    percented_patches_second(ax, percents)
    plt.show()


def plot_missing_data(data_frame, top_n=20):
    """
    Plots the count of missing values for each feature in the DataFrame.
    Args:
        data_frame (pd.DataFrame): The DataFrame to analyze.
        top_n (int): Number of top features to display (default is 20).
    """
    index_missings = data_frame.isna().sum().index
    missing_count = data_frame.isna().sum()
    missing_perc = data_frame.isna().sum() / len(data_frame) * 100

    ind_missing_count = list(zip(index_missings, missing_count, missing_perc))
    ind_missing_count.sort(key=lambda x: x[1], reverse=True)

    rep_missings = [
        (missing[0], missing[1], missing[2])
        for missing in ind_missing_count[:top_n]
        if missing[2] > 0
    ]
    rep_missings_df = pd.DataFrame(
        data=rep_missings, columns=["Feature", "Total", "PercentDF"]
    )

    plt.figure(figsize=(6, 2))
    ax = sns.barplot(y=rep_missings_df["Feature"], x=rep_missings_df["Total"])
    ax.set_title("Count of Missings")
    ax.set_ylabel("Feature Name")
    percented_patches_second(ax, rep_missings_df["PercentDF"].values)
    plt.show()


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

