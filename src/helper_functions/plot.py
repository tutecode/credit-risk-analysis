import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tabulate import tabulate

# Setting the style
sns.set_theme(style="ticks", palette="pastel")
sns.set(font_scale=0.8)

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


def plot_income_by_other_column(data_frame, income_colname, other_colname, top_n=30):
    """
    Plots the income column grouped by another column in the DataFrame.
    Args:
        data_frame (pd.DataFrame): The DataFrame to analyze.
        income_colname (str): Name of the income column.
        other_colname (str): Name of the other column to group by.
        top_n (int): Number of top rows to include in the plot (default is 30).
    """
    df_order_by_income = data_frame.sort_values(
        by=[income_colname], ascending=False, ignore_index=True
    )
    df_income_plot = df_order_by_income.loc[:top_n, [income_colname, other_colname]]

    plt.figure(figsize=(15, 4))
    plt.title(income_colname + " Grouped By " + other_colname)
    sns.barplot(x=df_income_plot[other_colname], y=df_income_plot[income_colname])
    plt.tight_layout()
    plt.show()

# function to plot distribution of numerical feature
def plotting_distribution_bar(df1, col_name, target_col="TARGET_LABEL_BAD=1"):

    fig, axes = plt.subplots(1, 1, figsize=(8, 3))
    fig.suptitle("Distribution of " + col_name)
    fig.align_labels()

    if df1[col_name].var() != 0:
        sns.countplot(ax=axes, data=df1, x=col_name, hue=target_col)
    else:
        sns.histplot(ax=axes, data=df1, x=col_name, hue=target_col, fill=True)

    # sns.boxplot(ax=axes[1], data=df1, y=target_col, x=col_name, orient="h")

    axes.set_ylabel("Target")
    # Establece los límites del eje y en 0 y 100 y la ubicación de los ticks en incrementos de 10.
    plt.yticks(range(0, 25000, 5000))

    # Establece los límites del eje x en 0 y 90 y la ubicación de los ticks cada 5.
    # plt.xticks(range(0, 8, 1))

    # Agrega una leyenda con las etiquetas y los valores de la media y la mediana.
    plt.legend()
    # axes[0].set_ylabel('Train')
    plt.tight_layout()
    plt.show()


def plotting_distribution(df1, col_name, target_col="TARGET_LABEL_BAD=1"):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    fig.suptitle("Distribution of " + col_name)
    fig.align_labels()

    if df1[col_name].var() != 0:
        sns.kdeplot(ax=axes[0], data=df1, x=col_name, hue=target_col, fill=True)
    else:
        sns.histplot(ax=axes[0], data=df1, x=col_name, hue=target_col, fill=True)

    sns.boxplot(ax=axes[1], data=df1, y=target_col, x=col_name, orient="h")
    axes[1].set_ylabel("Target")
    axes[0].set_ylabel('Train')
    plt.tight_layout()
    plt.show()


# # function to plot distribution of numerical feature
# def plotting_distribution(col_name, df, target_col):
#     fig, axes = plt.subplots(1, 2, figsize=(8, 3))
#     fig.suptitle("Distribution of " + col_name)
#     fig.align_labels()

#     if df[col_name].var() != 0:
#         sns.kdeplot(ax=axes[0], data=df, x=col_name, hue=target_col, fill=True)
#     else:
#         sns.histplot(ax=axes[0], data=df, x=col_name, hue=target_col, fill=True)

#     sns.boxplot(ax=axes[1], data=df, y=target_col, x=col_name, orient="h")
#     axes[1].set_ylabel("Target")
#     axes[0].set_ylabel('Train')
#     plt.tight_layout()
#     plt.show()


# # plotting counting values of categorical columns
# def plot_value_counts(fig, axes, col_name, train, col_percents, target_col):
#     fig.suptitle("Value Counts of " + col_name)
#     fig.align_labels()
#     sns.countplot(
#         ax=axes[0],
#         data=train,
#         y=col_name,
#         hue=target_col,
#         palette=sns.color_palette("ch:s=-.2,r=.6", n_colors=5),
#     )

#     percented_patches_second(axes[0], col_percents)
#     sns.color_palette("rocket", as_cmap=True)
#     sns.countplot(
#         ax=axes[1],
#         # data=test,
#         y=col_name,
#         palette=sns.color_palette("rocket_r", n_colors=5),
#     )
#     axes[0].set_xlabel("Train")
#     # axes[1].set_xlabel("Test")
#     axes[0].legend(
#         title="Target", labels=["Bad", "Good"], loc="upper left", bbox_to_anchor=(1, 1)
#     )
#     axes[0].set_ylabel("")
#     # axes[1].set_ylabel("")
#     plt.tight_layout()
#     plt.show()
#     plt.close(fig)

# plotting counting values of categorical columns
def plot_value_counts(df, col_name,  target_col="TARGET_LABEL_BAD=1"):
    fig,axes = plt.subplots(1,1, figsize=(8,3))
    fig.suptitle("Value Counts of " + col_name)
    fig.align_labels()
    sns.countplot(
        ax=axes,
        data=df,
        y=col_name,
        hue=target_col,
        palette=sns.color_palette("ch:s=-.2,r=.6", n_colors=5),
    )

# # plotting counting values of categorical columns
# def plot_value_counts(col_name, df, target_col):
#     fig,axes = plt.subplots(1,1, figsize=(8,2))
#     fig.suptitle("Value Counts of " + col_name)
#     fig.align_labels()
#     sns.countplot(
#         ax=axes,
#         data=df,
#         y=col_name,
#         hue=target_col,
#         palette=sns.color_palette("ch:s=-.2,r=.6", n_colors=5),
#     )

def plot_value_counts_big(col_name, df, target_col):
    fig,axes = plt.subplots(1,1, figsize=(8,8))
    fig.suptitle("Value Counts of " + col_name)
    fig.align_labels()
    sns.countplot(
        ax=axes,
        data=df,
        y=col_name,
        hue=target_col,
        palette=sns.color_palette("ch:s=-.2,r=.6", n_colors=5),
    )
    # Pending, to show percents

def plot_number_columns_type(df1, df2):
    # show number of columns per data type
    number_fields = len(df1.select_dtypes(include="number").columns)
    object_fields = len(df1.select_dtypes(exclude="number").columns)
    number_fields_o = len(df2.select_dtypes(include="number").columns)
    object_fields_o = len(df2.select_dtypes(exclude="number").columns)

    d = {
        "var_type": ["number", "object"],
        "quantity_cop": [number_fields, object_fields],
        "quantity_ori": [number_fields_o, object_fields_o],
    }

    quant_kind_vars = pd.DataFrame(data=d, index=[1, 2])

    fig, axes = plt.subplots(1, 2, figsize=(8, 2), sharey=True)
    fig.suptitle("Type columns Quantity Before and After Cleaning")
    fig.align_labels()
    sns.barplot(ax=axes[0], y=quant_kind_vars["var_type"], x=quant_kind_vars["quantity_ori"])
    sns.barplot(ax=axes[1], y=quant_kind_vars["var_type"], x=quant_kind_vars["quantity_cop"])

    axes[0].set_title("Number of Features")
    axes[0].set_xlabel("Original Dataset")
    axes[1].set_xlabel("New Dataset")

    target_dist = compute_stats_count(quant_kind_vars, "quantity_cop")
    percents_cop = [i[2] for i in target_dist]
    percented_patches_second(axes[1], percents_cop)

    target_dist = compute_stats_count(quant_kind_vars, "quantity_ori")
    percents_ori = [i[2] for i in target_dist]
    percented_patches_second(axes[0], percents_ori)

    plt.tight_layout()

    plt.show()