import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tabulate import tabulate

from helper_functions import preprocessing


# Setting the style
sns.set_theme(style="ticks", palette="pastel")
sns.set(font_scale=0.8)


def compute_stats_count(train, field, counting=False):
    """
    Compute statistics for counting occurrences of values in a given field of the provided DataFrame.

    Args:
    - train (pd.DataFrame): DataFrame containing the data.
    - field (str): The column name for which statistics are computed.
    - counting (bool, optional): If True, computes count and percentage of occurrences.
                                 If False, computes raw values and percentage of contribution.
                                 Defaults to False.

    Returns:
    - list of tuples: A list of tuples containing (value, count, percentage) statistics.
    """
    count, index, perc = [], [], []

    if counting:
        # Compute count of occurrences for each unique value in the specified field
        count = train[field].value_counts().values
        index = train[field].value_counts().index
        perc = train[field].value_counts().values / len(train[field]) * 100
    else:
        # Use raw values of the specified field
        count = train[field].values
        index = train[field].index
        perc = train[field].values / train[field].sum() * 100

    # Combine the computed statistics into a list of tuples
    stats_list = list(zip(index, count, perc))

    return stats_list


# percent patches type 1
def percented_patches_first(ax, col_stats):
    patches = ax.patches
    for i in range(len(patches)):
        x = patches[i].get_y() + patches[i].get_height() / 2
        y = patches[i].get_width() + 0.05
        ax.annotate("{:.2f}%".format(col_stats[i][2]), (y, x), va="center")


# percent patches type 2
def percented_patches_second(ax, col_stats):
    patches = ax.patches
    for i in range(len(patches)):
        x = patches[i].get_y() + patches[i].get_height() / 2
        y = patches[i].get_width() + 0.05
        ax.annotate("{:.2f}%".format(col_stats[i]), (y, x), va="center")


# function to split target = 0, and target =1, and obtain values for chart
def target_values(df, col_name, value, target_col="TARGET_LABEL_BAD=1"):
    if value < 0 or value > 1:
        return print("The nuber must be between 0 and 1")

    target = df[df[target_col] == value].sort_values(by=col_name).copy()
    counts = target[col_name].value_counts().sort_index()
    percentages = round(
        (target[col_name].value_counts(normalize=True).sort_index() * 100), 2
    )
    counts = target[col_name].value_counts().sort_index()
    max_value = counts.max()
    percentages = round(
        (target[col_name].value_counts(normalize=True).sort_index() * 100), 2
    )
    percentages_dic = percentages.to_dict()

    return (target, counts, max_value, percentages_dic)


# to show unique values for the columns
def plot_unique_value_counts(data_frame, top_n=10):
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


# shows distribution of missing data
def plot_missing_data(data_frame, top_n=20):
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


# shows a table of missing data
def display_missing_values(data_frame):
    missing_counts = data_frame.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    missing_data = pd.DataFrame(missing_counts, columns=["Missing Count"])
    missing_data.index.name = "Column"

    print(tabulate(missing_data, headers="keys", tablefmt="pretty"))


# plot distribution of the target variable
def plot_target_variable_distribution(data_frame, target_colname):
    target_dist = compute_stats_count(data_frame, target_colname, counting=True)
    target_df = pd.DataFrame(target_dist, columns=["Value", "Count", "Percentage"])
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


# shows the count of the different columns types
def plot_data_type_counts(data_frame):
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


# plot income vs city, it can be changed.
def plot_income_by_other_column(data_frame, income_colname, other_colname, top_n=30):
    df_order_by_income = data_frame.sort_values(
        by=[income_colname], ascending=False, ignore_index=True
    )
    df_income_plot = df_order_by_income.loc[:top_n, [income_colname, other_colname]]

    plt.figure(figsize=(15, 4))
    plt.title(income_colname + " Grouped By " + other_colname)
    sns.barplot(x=df_income_plot[other_colname], y=df_income_plot[income_colname])
    plt.tight_layout()

    plt.show()


# plotting counting values of categorical columns
def plot_value_counts(df, col_name, target_col="TARGET_LABEL_BAD=1"):
    fig, axes = plt.subplots(1, 1, figsize=(9, 4))
    fig.suptitle("Value Counts of " + col_name)
    fig.align_labels()
    sns.countplot(
        ax=axes,
        data=df,
        y=col_name,
        hue=target_col,
        palette=sns.color_palette("ch:s=-.2,r=.6", n_colors=5),
    )


# to plot the distribution, it can ver in vertical orientation or horizontal orientation
def plotting_distribution_bar(
    df, col_name, orientation="vertical", ordered=True, target_col="TARGET_LABEL_BAD=1"
):
    # chart's orientation
    if orientation == "horizontal":
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    elif orientation == "vertical":
        fig, axes = plt.subplots(2, 1, figsize=(10, 4))
    else:
        return print('unvalid orientation, must be "vertical" or "horizontal"')

    fig.suptitle("Distribution of " + col_name)
    fig.align_labels()

    # target = 0
    df_0, counts_0, max_value_0, percentages_0 = target_values(df, col_name, 0)
    sns.countplot(ax=axes[0], data=df_0, x=col_name, color="blue")

    # Add percentage labels above the bars with a small vertical shift
    x_position = 0
    for key, val in percentages_0.items():
        axes[0].text(
            x_position,
            counts_0[key] + max_value_0 * 0.02,
            f"{val}%",
            ha="center",
            va="bottom",
        )
        x_position += 1

    # target = 1
    df_1, counts_1, max_value_1, percentages_1 = target_values(df, col_name, 1)
    sns.countplot(ax=axes[1], data=df_1, x=col_name, color="red")

    # Add percentage labels above the bars with a small vertical shift
    x_position = 0
    for key, val in percentages_1.items():
        axes[1].text(
            x_position,
            counts_1[key] + max_value_1 * 0.02,
            f"{val}%",
            ha="center",
            va="bottom",
        )
        x_position += 1

    y_max = max(max_value_0, max_value_1)

    axes[0].set_ylabel("APPROVED")
    axes[1].set_ylabel("NOT APPROVED")
    axes[0].set_ylim(top=max_value_0 * 1.15)
    axes[1].set_ylim(top=max_value_1 * 1.15)
    plt.tight_layout()

    plt.show()


# function to plot distribution of numerical feature
def plotting_distribution_bar_double(df1, col_name, target_col="TARGET_LABEL_BAD=1"):
    fig, axes = plt.subplots(1, 1, figsize=(8, 3))
    fig.suptitle("Distribution of " + col_name)
    fig.align_labels()

    # if df1[col_name].var() != 0:
    sns.countplot(ax=axes, data=df1, x=col_name, hue=target_col)
    # else:
    #     sns.histplot(ax=axes, data=df1, x=col_name, hue=target_col, fill=True)

    # sns.boxplot(ax=axes[1], data=df1, y=target_col, x=col_name, orient="h")

    axes.set_ylabel("Target")
    # Establece los límites del eje y en 0 y 100 y la ubicación de los ticks en incrementos de 10.
    plt.yticks(range(0, 25000, 5000))

    # Establece los límites del eje x en 0 y 90 y la ubicación de los ticks cada 5.
    # plt.xticks(range(0, 8, 1))

    plt.legend()
    plt.tight_layout()
    plt.show()


#
# function to plot distribution of numerical feature
def plotting_distribution_kde(
    df, col_name, orientation="vertical", target_col="TARGET_LABEL_BAD=1"
):
    # chart's orientation
    if orientation == "horizontal":
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    elif orientation == "vertical":
        fig, axes = plt.subplots(2, 1, figsize=(10, 4))
    else:
        return print('unvalid orientation, must be "vertical" or "horizontal"')

    # fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    fig.suptitle("Distribution of " + col_name)
    fig.align_labels()

    df_0, _, max_value_0, _ = target_values(df, col_name, 0)
    sns.kdeplot(ax=axes[0], data=df_0, x=col_name, color="blue", fill=True)

    df_1, _, max_value_1, _ = target_values(df, col_name, 1)
    sns.kdeplot(ax=axes[1], data=df_1, x=col_name, color="red", fill=True)

    y_max = max(max_value_0, max_value_1)

    axes[0].set_ylabel("APPROVED")
    axes[1].set_ylabel("NOT APPROVED")
    plt.tight_layout()
    plt.show()


# small charts for cards only
def plotting_distribution_cards(
    df, col_name, orientation="vertical", ordered=True, target_col="TARGET_LABEL_BAD=1"
):
    # chart's orientation
    if orientation == "horizontal":
        fig, axes = plt.subplots(1, 2, figsize=(10, 2))
    elif orientation == "vertical":
        fig, axes = plt.subplots(2, 1, figsize=(10, 2))
    else:
        return print('unvalid orientation, must be "vertical" or "horizontal"')

    fig.suptitle("Distribution of " + col_name)
    fig.align_labels()

    # target = 0
    df_0, counts_0, max_value_0, percentages_0 = target_values(df, col_name, 0)
    sns.countplot(ax=axes[0], data=df_0, x=col_name, color="blue")

    # Add percentage labels above the bars with a small vertical shift
    x_position = 0
    for key, val in percentages_0.items():
        axes[0].text(
            x_position,
            counts_0[key] + max_value_0 * 0.02,
            f"{val}%",
            ha="center",
            va="bottom",
        )
        x_position += 1

    # target = 1
    df_1, counts_1, max_value_1, percentages_1 = target_values(df, col_name, 1)
    sns.countplot(ax=axes[1], data=df_1, x=col_name, color="red")

    # Add percentage labels above the bars with a small vertical shift
    x_position = 0
    for key, val in percentages_1.items():
        axes[1].text(
            x_position,
            counts_1[key] + max_value_1 * 0.02,
            f"{val}%",
            ha="center",
            va="bottom",
        )
        x_position += 1

    y_max = max(max_value_0, max_value_1)

    axes[0].set_ylabel("APPROVED")
    axes[1].set_ylabel("NOT APPROVED")
    axes[0].set_ylim(top=max_value_0 * 1.15)
    axes[1].set_ylim(top=max_value_1 * 1.15)
    plt.tight_layout()

    plt.show()
