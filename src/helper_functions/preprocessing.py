import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer


# # Setting the style
# sns.set_theme(style="ticks", palette="pastel")
# sns.set(font_scale=0.8)

# Function to change te repated column name
def repeated_name(df1, df2):
    metadata = df2

    meta_cols = metadata["Var_Title"].to_list()
    meta_cols[43] = "MATE_EDUCATION_LEVEL"

    # Set the new column to the train_data and test_data
    df1.columns = meta_cols
    return df1

# shows only numerical columns
def unique_numerical(df1, df2):
    print("{:<32}{:<15}{}\n".format("Feature Name", "UniqueCount", "RangeMeta"))
    number_field_names = df1.select_dtypes("number").columns.to_list()
    metadata = df2
    metadata_dic = {colname: idx for idx, colname in enumerate(df1.columns)}

    for number_field in number_field_names:
        # print(number_field.unique())
        print(
            "{:<32}{:<15}{}".format(
                number_field,
                len(df1[number_field].unique()),
                metadata.iloc[metadata_dic[number_field], 2],
            )
        )

    
# shows only non numerical columns
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


# for columns with lots of outliers
def proc_outliers(df, field):
    # impute nans with mean value of column
    df[field].replace({np.nan: df[field].mean()}, inplace=True)


# function for normalizing data at once
def normalized_data(df):
    df_cop = df.copy()
    target_col = "TARGET_LABEL_BAD=1"

    # 'PAYMENT_DAY': category = ["1 - 15", "16 - 30"]
    df_cop['PAYMENT_DAY'] = np.where(df_cop['PAYMENT_DAY'] <= 14, "1 - 14", "15 - 30")


    # 'MARITAL_STATUS': category =  {1:'single', 2:'married', 3:'other'}
    df_cop['MARITAL_STATUS'] = np.where(df_cop['MARITAL_STATUS'] == 1, "single",
                np.where(df_cop['MARITAL_STATUS'] == 2, "married", "other"))


    # 'QUANT_DEPENDANTS': numerical changes = [0, 1, 2, + 3]
    df_cop.loc[df_cop['QUANT_DEPENDANTS'] > 3, 'QUANT_DEPENDANTS'] = 3
    # 'HAS_DEPENDANTS': categorical column = {0:False, >0:True}
    df_cop['HAS_DEPENDANTS'] = np.where(df_cop['QUANT_DEPENDANTS'] >= 1, True, False)
    df_cop['HAS_DEPENDANTS'] =  df_cop['HAS_DEPENDANTS'].astype('boolean')

    # "RESIDENCE_TYPE": numerical changes = {1: 'owned', 2:'mortgage', 3:'rented', 4:'family', 5:'other'}
    imp_const_zero = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
    df_cop["RESIDENCE_TYPE"] = imp_const_zero.fit_transform(df_cop[["RESIDENCE_TYPE"]]).ravel()
    # categorical changes
    # mapping = {1: "owned", 2: "mortgage", 3: "rented", 4: "family", 5: "other"}
    df_cop["HAS_RESIDENCE"] = np.where(df_cop["RESIDENCE_TYPE"] == 1, True, False)
    df_cop['HAS_RESIDENCE'] =  df_cop['HAS_RESIDENCE'].astype('boolean')

    # "MONTHS_IN_RESIDENCE": category = ['0 - 6 months', '< 1 year', '+ 1 year']
    df_cop["MONTHS_IN_RESIDENCE"] = np.where(df_cop["MONTHS_IN_RESIDENCE"] <= 6, '0 - 6 months',
            np.where(df_cop["MONTHS_IN_RESIDENCE"] <= 12, '6 months - 1 year', '+ 1 year'))


    # "MONTHLY_INCOMES_TOT" and "OTHER_INCOMES" changed by "OTHER_INCOMES"
    # added to personal income in order to increase people who has less than minimal salary
    df_cop["MONTHLY_INCOMES_TOT"] = (df_cop["PERSONAL_MONTHLY_INCOME"] + df_cop["OTHER_INCOMES"])

    df_cop["MONTHLY_INCOMES_TOT"] = pd.cut(df_cop["MONTHLY_INCOMES_TOT"],
                bins=[0, 650, 1320, 3323, 8560, float('inf')],
                labels=['[0 - 650]', '[650 - 1320]', '[1320 - 3323]', '[3323 - 8560]', '[> 8560]'],
                right=False)


    # 'HAS_CARDS' category, replaces all cards.
    list_cards = ["FLAG_VISA", "FLAG_MASTERCARD", "FLAG_DINERS", "FLAG_AMERICAN_EXPRESS", "FLAG_OTHER_CARDS"]
    df_cop['HAS_CARDS'] = np.where(df_cop[list_cards].any(axis=1), True, False)
    df_cop['HAS_CARDS'] =  df_cop['HAS_CARDS'].astype('boolean')
    

    # "QUANT_BANKING_ACCOUNTS" and "QUANT_SPECIAL_BANKING_ACCOUNTS" changed by "HAS_BANKING_ACCOUNTS"
    # added to personal income in order to increase people who has less than minimal salary
    df_cop["QUANT_BANKING_ACCOUNTS"] = (df_cop["QUANT_BANKING_ACCOUNTS"] + df_cop["QUANT_SPECIAL_BANKING_ACCOUNTS"])    
    
    # 'HAS_BANKING_ACCOUNTS' category, replaces all accounts.
    df_cop["HAS_BANKING_ACCOUNTS"] = np.where(df_cop["QUANT_BANKING_ACCOUNTS"] == 0, False, True)
    df_cop['HAS_BANKING_ACCOUNTS'] =  df_cop['HAS_BANKING_ACCOUNTS'].astype('boolean')


    # 'PERSONAL_ASSETS_VALUE': changed to 'HAS_PERSONAL_ASSETS' = N, Y
    df_cop['HAS_PERSONAL_ASSETS'] = np.where(df_cop['PERSONAL_ASSETS_VALUE'] > 0, True, False)
    df_cop['HAS_PERSONAL_ASSETS'] =  df_cop['HAS_PERSONAL_ASSETS'].astype('boolean')
    

    # 'QUANT_CARS':changed to 'HAS_CARS' = N, Y
    df_cop['HAS_CARS'] = np.where(df_cop['QUANT_CARS'] == 0, False, True)
    df_cop["HAS_CARS"] = df_cop["HAS_CARS"].astype('boolean')

    # 'FLAG_EMAIL':changed to = N, Y
    df_cop['FLAG_EMAIL'] = np.where(df_cop['FLAG_EMAIL'] == 0, False, True)
    df_cop['FLAG_EMAIL'] = df_cop['FLAG_EMAIL'].astype('boolean')

    # 'FLAG_RESIDENCIAL_PHONE':changed to = N, Y
    df_cop['FLAG_RESIDENCIAL_PHONE'] = np.where(df_cop['FLAG_RESIDENCIAL_PHONE'] == "N", False, True)
    df_cop['FLAG_RESIDENCIAL_PHONE'] = df_cop['FLAG_RESIDENCIAL_PHONE'].astype('boolean')
    

    # 'FLAG_PROFESSIONAL_PHONE':changed to = N, Y
    df_cop['FLAG_PROFESSIONAL_PHONE'] = np.where(df_cop['FLAG_PROFESSIONAL_PHONE'] == "N", False, True)
    df_cop['FLAG_PROFESSIONAL_PHONE'] = df_cop['FLAG_PROFESSIONAL_PHONE'].astype('boolean')

    # COMPANY
    df_cop['COMPANY'] = np.where(df_cop['COMPANY'] == "N", False, True)
    df_cop['COMPANY'] = df_cop['COMPANY'].astype('boolean')
    
    # 'PRODUCT':changed to [1,2,7], PR1, PR2, PR7
    df_cop['PRODUCT'] = np.where(df_cop['PRODUCT'] == 1, "PR1", np.where(df_cop['PRODUCT'] == 2,"PR2","PR7"))

    # "APPLICATION_SUBMISSION_TYPE": 0 values changed to "Carga"
    df_cop.loc[df_cop["APPLICATION_SUBMISSION_TYPE"] != "Web", "APPLICATION_SUBMISSION_TYPE"] = "Carga"

    
    # 'SEX': deleted unknown values, changed to categorical
    df_cop.drop(df_cop[(df_cop["SEX"] == "N")].index,inplace=True,)
    df_cop.drop(df_cop[(df_cop["SEX"] == " ")].index,inplace=True,)
    
    
    # 'AGE'
    bins = [0, 18, 25, 35, 45, 60, float('inf')]
    labels = ['< 18', '18 - 25', '26 - 35', '36 - 45', '46 - 60', '> 60']
    df_cop['AGE'] = pd.cut(df_cop['AGE'], bins=bins, labels=labels) 

    return (df_cop, target_col)

def categorical_columns(df):
    # change columns to category, except boolean columns
    object_columns = [col for col in df.columns if df[col].dtype != 'boolean']
    df[object_columns] = df[object_columns].astype('category')
    return df


def delete_columns(df):

    # delete columns with 1 single value
    num_unique_values = df.nunique()
    columns_to_drop = num_unique_values[num_unique_values == 1].index
    df.drop(columns=columns_to_drop, inplace=True)

    # delete columns according to our criteria
    drop_columns=['ID_CLIENT', # index 
                'POSTAL_ADDRESS_TYPE', # not valid proportion
                'QUANT_DEPENDANTS',  # delete??
                # 'HAS_DEPENDANTS', # delete??
                'STATE_OF_BIRTH', # too many null values
                'CITY_OF_BIRTH', # too many values
                'NACIONALITY', # not valid proportion
                # RESIDENCIAL_STATE', # delete??
                'RESIDENCIAL_CITY', # too many unique values
                'RESIDENCIAL_BOROUGH', # too many unique values
                "RESIDENCIAL_PHONE_AREA_CODE", # too many unique values
                # 'FLAG_RESIDENCIAL_PHONE', # DELETE? if not chart
                "RESIDENCE_TYPE", # changed by HAS_RESIDENCE
                # "HAS_RESIDENCE", # DELETE?
                # "FLAG_EMAIL", # DELETE? if not chart
                "PERSONAL_MONTHLY_INCOME", # changed by 'MONTHLY_INCOMES_TOT'
                'OTHER_INCOMES', # changed by 'MONTHLY_INCOMES_TOT'
                'FLAG_VISA', # replaced by 'HAS_CARDS'
                'FLAG_MASTERCARD', # replaced by 'HAS_CARDS'
                'FLAG_DINERS', # replaced by 'HAS_CARDS'
                'FLAG_AMERICAN_EXPRESS', # replaced by 'HAS_CARDS'
                'FLAG_OTHER_CARDS', # replaced by 'HAS_CARDS'
                'QUANT_BANKING_ACCOUNTS', # replaced by 'HAS_BANKING_ACCOUNTS'
                'QUANT_SPECIAL_BANKING_ACCOUNTS', # replaced by 'HAS_BANKING_ACCOUNTS'
                'PERSONAL_ASSETS_VALUE', # replaced by 'HAS_PERSONAL_ASSETS'
                'QUANT_CARS', # replaced by 'HAS_CARS'
                'PROFESSIONAL_STATE', # more than 60% of empty values
                'PROFESSIONAL_CITY', # too many different values
                'PROFESSIONAL_BOROUGH', # too many different values
                'PROFESSIONAL_PHONE_AREA_CODE', # more than 60% of empty values
                'MONTHS_IN_THE_JOB', # more than 95% of 0 as a value
                'PROFESSION_CODE', # not enough information, over 7k null values
                'OCCUPATION_TYPE', # not enough information, over 7k null values
                'MATE_PROFESSION_CODE', # over 50% of empty values
                'MATE_EDUCATION_LEVEL', # over 60% of empty values
                # 'PRODUCT', delete?, 3 different values, 
                'RESIDENCIAL_ZIP_3', # too many unique values'
                'PROFESSIONAL_ZIP_3'] # too many unique values'
    
    list_not_find = []
    list_removed = []
    
    for outside_column in drop_columns:
        if(outside_column in df.columns):
            list_removed.append(outside_column)
            df.drop(columns = outside_column, axis=1, inplace=True)
        else:
            list_not_find.append(outside_column)


    print("Those columns were removed: \n",list_removed)
    print("\nThose columns were not found: \n",list_not_find)

    return df





    # # Ahora el DataFrame df ya no contiene las columnas con un solo valor
    # print(df)

# def trunc(valor, liminf, limsup):
#     if valor < liminf:
#         return liminf
#     if valor > limsup:
#         return limsup
#     if valor > liminf and valor < limsup:
#         return valor


# def compute_limits(df, field):
#     q1 = np.percentile(df[field], 25)
#     q3 = np.percentile(df[field], 75)
#     iqr = q3 - q1
#     limsup = q3 + 1.5 * iqr
#     liminf = q1 - 1.5 * iqr
#     return liminf, limsup

# # pasar a plot?
# def plot_outliers(df, field):
#     fig, axs = plt.subplots(1, 2, figsize=(10, 2))
#     sns.boxplot(x=df[field], ax=axs[0])
#     sns.boxplot(x=df["new" + field], ax=axs[1])


# def proc_outliers(df, field):
#     # impute nans with mean value of column
#     df[field].replace({np.nan: df[field].mean()}, inplace=True)

#     # compute quantiles
#     liminf, limsup = compute_limits(df, field)

#     # apply truncated function
#     df["new" + field] = df[field].apply(lambda val: trunc(val, liminf, limsup))

#     # plot before and after of correct outliers
#     plot_outliers(df, field)

#     # update dataframe
#     df[field] = df["new" + field]
#     df.drop(["new" + field], axis=1, inplace=True)


# # function to cast numerical/object to category feature
# def cast_to_category(col_name, train):
#     train[col_name] = train[col_name].astype("category")
#     # test[col_name] = test[col_name].astype("category")
#     print(col_name + ": " + str(train[col_name].dtype))
#     # print("datatype train of " + col_name + ": " + str(train[col_name].dtype))
#     # print("datatype test of " + col_name + ": " + str(test[col_name].dtype))


# # create temporal column on dataframe
# def create_tmp_column(col_name, train):
#     # copy current column to temporal column
#     temp_col = col_name + "_tmp"
#     train[temp_col] = train[col_name]
#     return temp_col


# # remove temporal column on dataframe
# def remove_tmp_column(col_name, train):
#     train.drop([col_name], axis=1, inplace=True)
#     # test.drop([col_name], axis=1, inplace=True)


# # get percentages by target label
# def get_percents_by_target(col_name, train, target_col):
#     target_good = train[train[target_col] == 0][col_name]
#     target_bad = train[train[target_col] == 1][col_name]
#     good_vars = np.array(target_good.value_counts().values)
#     bad_vars = np.array(target_bad.value_counts().values)

#     matrix_targets = np.vstack((good_vars, bad_vars))
#     matrix_targets_sum = np.sum(matrix_targets, axis=0)
#     # good_sum = np.sum(good_vars) * np.ones((3))
#     # bad_sum = np.sum(bad_vars) * np.ones((3))

#     lenght_cat_vars = len(good_vars)
#     percents_labels = []
#     if lenght_cat_vars != 1:
#         good_perc = good_vars / matrix_targets_sum
#         bad_perc = bad_vars / matrix_targets_sum
#         percents_labels = np.append(good_perc, bad_perc)
#     else:
#         good_perc = target_good.value_counts().values / (
#             len(target_good) + len(target_bad)
#         )
#         bad_perc = target_bad.value_counts().values / (
#             len(target_good) + len(target_bad)
#         )
#         percents_labels.append(good_perc[0])
#         percents_labels.append(bad_perc[0])
#     return percents_labels


# # helper functions
# def compute_limits(df, field):
#     q1 = np.percentile(df[field], 25)
#     q3 = np.percentile(df[field], 75)
#     iqr = q3 - q1
#     limsup = q3 + 1.5 * iqr
#     liminf = q1 - 1.5 * iqr
#     print("lims: [{},{}]".format(liminf, limsup))
#     return liminf, limsup


# def plot_outliers(df, field, tmp_field):
#     fig, axs = plt.subplots(1, 2, figsize=(10, 2))
#     axs[1].set_title("Remove outliers from " + field)
#     sns.boxplot(x=df[field], ax=axs[0])
#     axs[0].set_title("Original outliers from " + field)
#     sns.boxplot(x=df[tmp_field], ax=axs[1])


# def proc_outliers(df, field):
#     # impute nans with mean value of column
#     df[field].replace({np.nan: df[field].mean()}, inplace=True)

#     # compute quantiles
#     liminf, limsup = compute_limits(df, field)

#     tmp_field = "new" + field
#     # apply truncated function
#     df[tmp_field] = df[field]
#     df.loc[df[tmp_field] < liminf, tmp_field] = liminf
#     df.loc[df[tmp_field] > limsup, tmp_field] = limsup

#     # plot before and after of correct outliers
#     plot_outliers(df, field, tmp_field)

#     # update dataframe
#     df[field] = df[tmp_field]
#     df.drop([tmp_field], axis=1, inplace=True)


# def repeated_name(df1, df2):
#     metadata = df2

#     meta_cols = metadata["Var_Title"].to_list()
#     meta_cols[43] = "MATE_EDUCATION_LEVEL"

#     # Set the new column to the train_data and test_data
#     df1.columns = meta_cols
#     # app_test.columns = meta_cols[:-1]
#     # print(df1.columns)
#     return df1
#     #app_train["MATE_EDUCATION_LEVEL"].info()
#     #app_test["MATE_EDUCATION_LEVEL"].info()


# # def unique_numerical(df1, df2):
# #     print("{:<32}{:<15}{}\n".format("Feature Name", "UniqueCount", "RangeMeta"))
# #     number_field_names = df1.select_dtypes("number").columns.to_list()
# #     metadata = df2
# #     metadata_dic = {colname: idx for idx, colname in enumerate(df1.columns)}

# #     for number_field in number_field_names:
# #         print(
# #             "{:<32}{:<15}{}".format(
# #                 number_field,
# #                 len(df1[number_field].unique()),
# #                 metadata.iloc[metadata_dic[number_field], 2],
# #             )
# #         )

# def unique_numerical(df1, df2):
#     print("{:<32}{:<15}{}\n".format("Feature Name", "UniqueCount", "RangeMeta"))
#     number_field_names = df1.select_dtypes("number").columns.to_list()
#     metadata = df2
#     metadata_dic = {colname: idx for idx, colname in enumerate(df1.columns)}
#     # metadata_dic = {colname: idx for idx, colname in enumerate(app_train.columns)}

#     for number_field in number_field_names:
#         # print(number_field.unique())
#         print(
#             "{:<32}{:<15}{}".format(
#                 number_field,
#                 len(df1[number_field].unique()),
#                 metadata.iloc[metadata_dic[number_field], 2],
#             )
#         )

# def unique_categorical(df1, df2):
#     category_field_names = df1.select_dtypes(exclude="number").columns.to_list()
#     metadata_dic = {colname: idx for idx, colname in enumerate(df1.columns)}
#     print("{:<32}{:<15}{}\n".format("Feature Name", "UniqueCount", "RangeMeta"))
#     for categorical_field in category_field_names:
#         print(
#             "{:<32}{:<15}{}".format(
#                 categorical_field,
#                 len(df1[categorical_field].unique()),
#                 df2.iloc[metadata_dic[categorical_field], 2],
#             )
#         )


# def move_target_end(df):
#     current_cols_train = df.columns.to_list()
#     idx_target = df.columns.to_list().index("TARGET_LABEL_BAD=1")
#     if(df.iloc[:,-1:].columns[0] != df.iloc[:,idx_target:].columns[0]):
#         features_cols = current_cols_train[:idx_target] + current_cols_train[idx_target+1:] + [current_cols_train[idx_target]]
#         #crear un nuevo df
#         df = df[features_cols]
#     else:
#         print("Target is the last column")

# #SET AT THE END
# def delete_columns(df):
    
#     df.drop(columns= ['ID_CLIENT'], inplace=True)