import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer


# Setting the style
sns.set_theme(style="ticks", palette="pastel")
sns.set(font_scale=0.8)


# function for normalizing data at once.
def normalized_data(df):
    df_cop = df.copy()
    target_col = "TARGET_LABEL_BAD=1"

    # 'PAYMENT_DAY': category = ["1 - 15", "16 - 30"]
    df_cop['PAYMENT_DAY'] = np.where(df_cop['PAYMENT_DAY'] <= 14, "1 - 14", "15 - 30")


    # 'MARITAL_STATUS': category =  {1:'single', 2:'married', 3:'other'}
    df_cop['MARITAL_STATUS'] = np.where(df_cop['MARITAL_STATUS'] == 1, "single",
                np.where(df_cop['MARITAL_STATUS'] == 2, "married", "Other"))
    df_cop['MARITAL_STATUS'] = df_cop['MARITAL_STATUS'].astype('category')


    # 'QUANT_DEPENDANTS': numerical changes = [0, 1, 2, + 3]
    df_cop.loc[df_cop['QUANT_DEPENDANTS'] > 3, 'QUANT_DEPENDANTS'] = 3
    # 'HAS_DEPENDANTS': categorical column = {0:False, >0:True}
    df_cop['HAS_DEPENDANTS'] = np.where(df_cop['QUANT_DEPENDANTS'] >= 1, True, False)


    # "RESIDENCE_TYPE": numerical changes = {1: 'owned', 2:'mortgage', 3:'rented', 4:'family', 5:'other'}
    imp_const_zero = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
    df_cop["RESIDENCE_TYPE"] = imp_const_zero.fit_transform(df_cop[["RESIDENCE_TYPE"]]).ravel()
    # categorical changes
    mapping = {1: "owned", 2: "mortgage", 3: "rented", 4: "family", 5: "other"}
    df_cop["RESIDENCE_TYPE"] = df_cop["RESIDENCE_TYPE"].map(lambda x: mapping[x] if x in mapping else "other")
    df_cop["RESIDENCE_TYPE"] = df_cop["RESIDENCE_TYPE"].astype('category')






    return (df_cop, target_col)

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

# pasar a plot?
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


# function to cast numerical/object to category feature
def cast_to_category(col_name, train):
    train[col_name] = train[col_name].astype("category")
    # test[col_name] = test[col_name].astype("category")
    print(col_name + ": " + str(train[col_name].dtype))
    # print("datatype train of " + col_name + ": " + str(train[col_name].dtype))
    # print("datatype test of " + col_name + ": " + str(test[col_name].dtype))


# create temporal column on dataframe
def create_tmp_column(col_name, train):
    # copy current column to temporal column
    temp_col = col_name + "_tmp"
    train[temp_col] = train[col_name]
    return temp_col


# remove temporal column on dataframe
def remove_tmp_column(col_name, train):
    train.drop([col_name], axis=1, inplace=True)
    # test.drop([col_name], axis=1, inplace=True)


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


# helper functions
def compute_limits(df, field):
    q1 = np.percentile(df[field], 25)
    q3 = np.percentile(df[field], 75)
    iqr = q3 - q1
    limsup = q3 + 1.5 * iqr
    liminf = q1 - 1.5 * iqr
    print("lims: [{},{}]".format(liminf, limsup))
    return liminf, limsup


def plot_outliers(df, field, tmp_field):
    fig, axs = plt.subplots(1, 2, figsize=(10, 2))
    axs[1].set_title("Remove outliers from " + field)
    sns.boxplot(x=df[field], ax=axs[0])
    axs[0].set_title("Original outliers from " + field)
    sns.boxplot(x=df[tmp_field], ax=axs[1])


def proc_outliers(df, field):
    # impute nans with mean value of column
    df[field].replace({np.nan: df[field].mean()}, inplace=True)

    # compute quantiles
    liminf, limsup = compute_limits(df, field)

    tmp_field = "new" + field
    # apply truncated function
    df[tmp_field] = df[field]
    df.loc[df[tmp_field] < liminf, tmp_field] = liminf
    df.loc[df[tmp_field] > limsup, tmp_field] = limsup

    # plot before and after of correct outliers
    plot_outliers(df, field, tmp_field)

    # update dataframe
    df[field] = df[tmp_field]
    df.drop([tmp_field], axis=1, inplace=True)


def repeated_name(df1, df2):
    metadata = df2

    meta_cols = metadata["Var_Title"].to_list()
    meta_cols[43] = "MATE_EDUCATION_LEVEL"

    # Set the new column to the train_data and test_data
    df1.columns = meta_cols
    # app_test.columns = meta_cols[:-1]
    # print(df1.columns)
    return df1
    #app_train["MATE_EDUCATION_LEVEL"].info()
    #app_test["MATE_EDUCATION_LEVEL"].info()


# def unique_numerical(df1, df2):
#     print("{:<32}{:<15}{}\n".format("Feature Name", "UniqueCount", "RangeMeta"))
#     number_field_names = df1.select_dtypes("number").columns.to_list()
#     metadata = df2
#     metadata_dic = {colname: idx for idx, colname in enumerate(df1.columns)}

#     for number_field in number_field_names:
#         print(
#             "{:<32}{:<15}{}".format(
#                 number_field,
#                 len(df1[number_field].unique()),
#                 metadata.iloc[metadata_dic[number_field], 2],
#             )
#         )

def unique_numerical(df1, df2):
    print("{:<32}{:<15}{}\n".format("Feature Name", "UniqueCount", "RangeMeta"))
    number_field_names = df1.select_dtypes("number").columns.to_list()
    metadata = df2
    metadata_dic = {colname: idx for idx, colname in enumerate(df1.columns)}
    # metadata_dic = {colname: idx for idx, colname in enumerate(app_train.columns)}

    for number_field in number_field_names:
        # print(number_field.unique())
        print(
            "{:<32}{:<15}{}".format(
                number_field,
                len(df1[number_field].unique()),
                metadata.iloc[metadata_dic[number_field], 2],
            )
        )

def unique_categorical(df1, df2):
    category_field_names = df1.select_dtypes(exclude="number").columns.to_list()
    metadata_dic = {colname: idx for idx, colname in enumerate(df1.columns)}
    print("{:<32}{:<15}{}\n".format("Feature Name", "UniqueCount", "RangeMeta"))
    for categorical_field in category_field_names:
        print(
            "{:<32}{:<15}{}".format(
                categorical_field,
                len(df1[categorical_field].unique()),
                df2.iloc[metadata_dic[categorical_field], 2],
            )
        )


def move_target_end(df):
    current_cols_train = df.columns.to_list()
    idx_target = df.columns.to_list().index("TARGET_LABEL_BAD=1")
    if(df.iloc[:,-1:].columns[0] != df.iloc[:,idx_target:].columns[0]):
        features_cols = current_cols_train[:idx_target] + current_cols_train[idx_target+1:] + [current_cols_train[idx_target]]
        #crear un nuevo df
        df = df[features_cols]
    else:
        print("Target is the last column")

#SET AT THE END
def delete_columns(df):
    
    df.drop(columns= ['ID_CLIENT'], inplace=True)