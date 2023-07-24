import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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




### Analyzing Numerical columns
# functions to treats outliers
def trunc(valor, liminf, limsup):
    if valor < liminf:
        return liminf
    if valor > limsup:
        return limsup
    if valor > liminf and valor < limsup:
        return valor


def compute_limits(df, field):
    q1 = np.percentile(df[field], 25)
    q3 = np.percentile(df[field], 75)
    iqr = q3 - q1
    limsup = q3 + 1.5 * iqr
    liminf = q1 - 1.5 * iqr
    return liminf, limsup


def plot_outliers(df, field):
    fig, axs = plt.subplots(1, 2, figsize=(10, 2))
    sns.boxplot(x=df[field], ax=axs[0])
    sns.boxplot(x=df["new" + field], ax=axs[1])


def proc_outliers(df, field):
    # impute nans with mean value of column
    df[field].replace({np.nan: df[field].mean()}, inplace=True)

    # compute quantiles
    liminf, limsup = compute_limits(df, field)

    # apply truncated function
    df["new" + field] = df[field].apply(lambda val: trunc(val, liminf, limsup))

    # plot before and after of correct outliers
    plot_outliers(df, field)

    # update dataframe
    df[field] = df["new" + field]
    df.drop(["new" + field], axis=1, inplace=True)

# function to plot distribution of numerical feature
def plotting_distribution(col_name, train, test, target_col):
    fig, axes = plt.subplots(2, 2, figsize=(8, 3))
    fig.suptitle("Distribution of " + col_name)
    fig.align_labels()

    if train[col_name].var() != 0:
        sns.kdeplot(ax=axes[0][0], data=train, x=col_name, hue=target_col, fill=True)
    else:
        sns.histplot(ax=axes[0][0], data=train, x=col_name, hue=target_col, fill=True)
    if test[col_name].var() != 0:
        sns.kdeplot(ax=axes[1][0], data=test, x=col_name, fill=True)
    else:
        sns.histplot(ax=axes[1][0], x=test.loc[:, col_name], fill=True)

    sns.boxplot(ax=axes[0][1], data=train, y=target_col, x=col_name, orient="h")
    sns.boxplot(ax=axes[1][1], data=test, x=col_name)
    axes[0][1].set_ylabel("Target")
    axes[0][0].legend(
        loc="upper left", title="Target", labels=["Bad", "Good"], bbox_to_anchor=(1, 1)
    )
    axes[0][0].set_ylabel("Train")
    axes[1][0].set_ylabel("Test")
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    
# plotting counting values of categorical columns
def plot_value_counts(fig, axes, col_name, train, test, col_percents, target_col):
    fig.suptitle("Value Counts of " + col_name)
    fig.align_labels()
    sns.countplot(
        ax=axes[0],
        data=train,
        y=col_name,
        hue=target_col,
        palette=sns.color_palette("ch:s=-.2,r=.6", n_colors=5),
    )

    plot.percented_patches_second(axes[0], col_percents)
    sns.color_palette("rocket", as_cmap=True)
    sns.countplot(
        ax=axes[1],
        data=test,
        y=col_name,
        palette=sns.color_palette("rocket_r", n_colors=5),
    )
    axes[0].set_xlabel("Train")
    axes[1].set_xlabel("Test")
    axes[0].legend(
        title="Target", labels=["Bad", "Good"], loc="upper left", bbox_to_anchor=(1, 1)
    )
    axes[0].set_ylabel("")
    axes[1].set_ylabel("")
    plt.tight_layout()
    plt.show()
    plt.close(fig)
# function to cast numerical/object to category feature
def cast_to_category(col_name, train, test):
    train[col_name] = train[col_name].astype("category")
    test[col_name] = test[col_name].astype("category")
    print("datatype train of " + col_name + ": " + str(train[col_name].dtype))
    print("datatype test of " + col_name + ": " + str(test[col_name].dtype))
# create temporal column on dataframe
def create_tmp_column(col_name, train, test):
    # copy current column to temporal column
    temp_col = col_name + "tmp"
    train[temp_col] = train[col_name]
    test[temp_col] = test[col_name]
    return temp_col
# remove temporal column on dataframe
def remove_tmp_column(col_name, train, test):
    train.drop([col_name], axis=1, inplace=True)
    test.drop([col_name], axis=1, inplace=True)
# get percentages by target label
def get_percents_by_target(col_name, train, target_col):
    target_good = train[train[target_col] == 0][col_name]
    target_bad = train[train[target_col] == 1][col_name]
    good_vars = np.array(target_good.value_counts().values)
    bad_vars = np.array(target_bad.value_counts().values)

    matrix_targets = np.vstack((good_vars, bad_vars))
    matrix_targets_sum = np.sum(matrix_targets, axis=0)
    # good_sum = np.sum(good_vars) * np.ones((3))
    # bad_sum = np.sum(bad_vars) * np.ones((3))

    lenght_cat_vars = len(good_vars)
    percents_labels = []
    if lenght_cat_vars != 1:
        good_perc = good_vars / matrix_targets_sum
        bad_perc = bad_vars / matrix_targets_sum
        percents_labels = np.append(good_perc, bad_perc)
    else:
        good_perc = target_good.value_counts().values / (
            len(target_good) + len(target_bad)
        )
        bad_perc = target_bad.value_counts().values / (
            len(target_good) + len(target_bad)
        )
        percents_labels.append(good_perc[0])
        percents_labels.append(bad_perc[0])
    return percents_labels

def barplot(data_df,x,y):
    plt.figure(figsize=(15, 4))
    plt.title(y + " By " + x)
    sns.barplot(data_df, x=x, y=y)
    plt.tight_layout()
    plt.show()