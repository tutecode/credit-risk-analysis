import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

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
    df_cop['PAYMENT_DAY'] = np.where(df_cop['PAYMENT_DAY'] <= 14, "1_14", "15_30")


    # 'MARITAL_STATUS': category =  {1:'single', 2:'married', 3:'other'}
    df_cop['MARITAL_STATUS'] = np.where(df_cop['MARITAL_STATUS'] == 1, "single",
                np.where(df_cop['MARITAL_STATUS'] == 2, "married", "other"))


    # 'QUANT_DEPENDANTS': numerical changes = [0, 1, 2, + 3]
    df_cop.loc[df_cop['QUANT_DEPENDANTS'] > 3, 'QUANT_DEPENDANTS'] = 3
    # 'HAS_DEPENDANTS': categorical column = {0:False, >0:True}
    df_cop['HAS_DEPENDANTS'] = np.where(df_cop['QUANT_DEPENDANTS'] >= 1, True, False)
    df_cop['HAS_DEPENDANTS'] =  df_cop['HAS_DEPENDANTS'].astype('bool')

    # "RESIDENCE_TYPE": numerical changes = {1: 'owned', 2:'mortgage', 3:'rented', 4:'family', 5:'other'}
    imp_const_zero = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
    df_cop["RESIDENCE_TYPE"] = imp_const_zero.fit_transform(df_cop[["RESIDENCE_TYPE"]]).ravel()
    # categorical changes
    # mapping = {1: "owned", 2: "mortgage", 3: "rented", 4: "family", 5: "other"}
    df_cop["HAS_RESIDENCE"] = np.where(df_cop["RESIDENCE_TYPE"] == 1, True, False)
    df_cop['HAS_RESIDENCE'] =  df_cop['HAS_RESIDENCE'].astype('bool')

    # "MONTHS_IN_RESIDENCE": category = ['0 - 6 months', '< 1 year', '+ 1 year']
    df_cop["MONTHS_IN_RESIDENCE"] = np.where(df_cop["MONTHS_IN_RESIDENCE"] <= 6, '0_6',
            np.where(df_cop["MONTHS_IN_RESIDENCE"] <= 12, '6_12', '>_12'))


    # "MONTHLY_INCOMES_TOT" and "OTHER_INCOMES" changed by "OTHER_INCOMES"
    # added to personal income in order to increase people who has less than minimal salary
    df_cop["MONTHLY_INCOMES_TOT"] = (df_cop["PERSONAL_MONTHLY_INCOME"] + df_cop["OTHER_INCOMES"])

    df_cop["MONTHLY_INCOMES_TOT"] = pd.cut(df_cop["MONTHLY_INCOMES_TOT"],
                bins=[0, 650, 1320, 3323, 8560, float('inf')],
                labels=['[0_650]', '[650_1320]', '[1320_3323]', '[3323_8560]', '[>8560]'],
                right=False)


    # 'HAS_CARDS' category, replaces all cards.
    list_cards = ["FLAG_VISA", "FLAG_MASTERCARD", "FLAG_DINERS", "FLAG_AMERICAN_EXPRESS", "FLAG_OTHER_CARDS"]
    df_cop['HAS_CARDS'] = np.where(df_cop[list_cards].any(axis=1), True, False)
    df_cop['HAS_CARDS'] =  df_cop['HAS_CARDS'].astype('bool')
    

    # "QUANT_BANKING_ACCOUNTS" and "QUANT_SPECIAL_BANKING_ACCOUNTS" changed by "HAS_BANKING_ACCOUNTS"
    # added to personal income in order to increase people who has less than minimal salary
    df_cop["QUANT_BANKING_ACCOUNTS"] = (df_cop["QUANT_BANKING_ACCOUNTS"] + df_cop["QUANT_SPECIAL_BANKING_ACCOUNTS"])    
    
    # 'HAS_BANKING_ACCOUNTS' category, replaces all accounts.
    df_cop["HAS_BANKING_ACCOUNTS"] = np.where(df_cop["QUANT_BANKING_ACCOUNTS"] == 0, False, True)
    df_cop['HAS_BANKING_ACCOUNTS'] =  df_cop['HAS_BANKING_ACCOUNTS'].astype('bool')


    # 'PERSONAL_ASSETS_VALUE': changed to 'HAS_PERSONAL_ASSETS' = N, Y
    df_cop['HAS_PERSONAL_ASSETS'] = np.where(df_cop['PERSONAL_ASSETS_VALUE'] > 0, True, False)
    df_cop['HAS_PERSONAL_ASSETS'] =  df_cop['HAS_PERSONAL_ASSETS'].astype('bool')
    

    # 'QUANT_CARS':changed to 'HAS_CARS' = N, Y
    df_cop['HAS_CARS'] = np.where(df_cop['QUANT_CARS'] == 0, False, True)
    df_cop["HAS_CARS"] = df_cop["HAS_CARS"].astype('bool')


    # "APPLICATION_SUBMISSION_TYPE": 0 values changed to "Carga"
    df_cop.loc[df_cop["APPLICATION_SUBMISSION_TYPE"] != "Web", "APPLICATION_SUBMISSION_TYPE"] = "Carga"

    
    # 'SEX': deleted unknown values, changed to categorical
    df_cop.drop(df_cop[(df_cop["SEX"] == "N")].index,inplace=True,)
    df_cop.drop(df_cop[(df_cop["SEX"] == " ")].index,inplace=True,)
    
    
    # 'AGE'
    bins = [0, 18, 25, 35, 45, 60, float('inf')]
    labels = ['<_18', '18_25', '26_35', '36_45', '46_60', '>_60']
    df_cop['AGE'] = pd.cut(df_cop['AGE'], bins=bins, labels=labels) 

    return (df_cop, target_col)

def categorical_columns(df):
    # change columns to category, except bool columns
    object_columns = [col for col in df.columns if df[col].dtype != 'bool']
    df[object_columns] = df[object_columns].astype('category')
    return df


def delete_columns(df):

    # delete columns with single values
    num_unique_values = df.nunique()
    columns_to_drop = num_unique_values[num_unique_values == 1].index
    df.drop(columns=columns_to_drop, inplace=True)

    # delete columns according to our criteria
    drop_columns=['ID_CLIENT', # index 
                'POSTAL_ADDRESS_TYPE', # not valid proportion
                # 'QUANT_DEPENDANTS',  # delete??
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
