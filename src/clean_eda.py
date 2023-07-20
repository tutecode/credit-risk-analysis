import numpy as np
import pandas as pd

def object_to_category(df):

    # This column has 2 valid values, 'Web' and 'Carga'
    # The 3rd value (which is not valid) will be replaced with 'Web'
    df['APPLICATION_SUBMISSION_TYPE'] = np.where(df['APPLICATION_SUBMISSION_TYPE']=='Web', 'Web', 'Carga')
    df['APPLICATION_SUBMISSION_TYPE'] = df['APPLICATION_SUBMISSION_TYPE'].astype('category')


    # This column has 2 valid vaues, 'M' and 'F', any other value will be replaced randomly.
    valid_values = ['M', 'F']  # las categorías válidas
    distribution = df[df['SEX'].isin(valid_values)]['SEX'].value_counts(normalize=True)
    out_of_category = ~df['SEX'].isin(valid_values) # Identify the non valid values

    # Change the non valid values to a valid value, converted to categorical
    df.loc[out_of_category, 'SEX'] = np.random.choice(distribution.index, size=len(df[out_of_category]), p=distribution.values)
    df['SEX'] = df['SEX'].astype('category')


    # # 'RESIDENCIAL_STATE' converted to categorical, has no missing values
    # df['RESIDENCIAL_STATE']=df['RESIDENCIAL_STATE'].astype('category')

    # 'FLAG_RESIDENCIAL_PHONE' converted to categorical, has no missing values
    df['FLAG_RESIDENCIAL_PHONE'] = df['FLAG_RESIDENCIAL_PHONE'].astype('category')

    #  'COMPANY' converted to categorical, has no missing values
    df['COMPANY'] = df['COMPANY'].astype('category')

    # 'FLAG_PROFESSIONAL_PHONE' converted to categorical, has no missing values
    df['FLAG_PROFESSIONAL_PHONE'] = df['FLAG_RESIDENCIAL_PHONE'].astype('category')

    try:
        # not relevant, lot's of missing values or too many different arguments
        df.drop(columns=['STATE_OF_BIRTH', # too many empty values
            'CITY_OF_BIRTH', 'RESIDENCIAL_CITY', 'RESIDENCIAL_BOROUGH', # too many unique values
            'RESIDENCIAL_PHONE_AREA_CODE', # too many empty values 
            'PROFESSIONAL_STATE', # too many empty values
            'PROFESSIONAL_CITY', # uncountable values to analize
            'PROFESSIONAL_BOROUGH', # too many empty values
            'PROFESSIONAL_PHONE_AREA_CODE', # too many empty values
            'RESIDENCIAL_ZIP_3', # uncountable values to analize
            'PROFESSIONAL_ZIP_3', # uncountable values to analize
            'RESIDENCIAL_STATE'], # was creating bias in the model
            inplace=True)
    except:
        print('Note: There are no columns to delete, it seems they have already been deleted')
    return(df)

def number_to_category(df):
    
    # 'PAYMENT_DAY': split every 15 days, converted to category
    df['PAYMENT_DAY_C'] = np.where(df['PAYMENT_DAY'] < 15, '1-14', '15-30')
    df['PAYMENT_DAY_C'] = df['PAYMENT_DAY_C'].astype('category')


    # 'POSTAL_ADDRESS_TYPE': changed to categorical, has no missing values
    df['POSTAL_ADDRESS_TYPE'] = df['POSTAL_ADDRESS_TYPE'].astype('category')


    # This column has 7 different values, values 1 and 2, are predominant, others added as a 3rd category
    marital_array = df['MARITAL_STATUS'].unique() # get all unique values
    marital_array = np.delete(marital_array, np.where(marital_array == 1)) # delete values = 1
    marital_array = np.delete(marital_array, np.where(marital_array == 2)) # delete values = 2

    # Values different to 1 and 2, added to 3
    df['MARITAL_STATUS'] = df['MARITAL_STATUS'].replace(marital_array, 3) # change all different values to 3
    df['MARITAL_STATUS'] = df['MARITAL_STATUS'].astype('category')


    # 'QUANT_DEPENDANTS': values, 0 for not dependants and 1 if have dependants, changed to categorical
    df['HAS_DEPENDANTS'] = [1 if dep > 0 else 0 for dep in df['QUANT_DEPENDANTS']] # 0 AND 1
    df['HAS_DEPENDANTS'] = df['HAS_DEPENDANTS'].astype('category')


    # 'NACIONALITY': Reduced to 2 types, 1 and 2 (adding 0 to 2), changed as category
    df['NACIONALITY'] = [1 if nac == 1 else 2 for nac in df['NACIONALITY']]
    df['NACIONALITY'] = df['NACIONALITY'].astype('category')


    # MONTHS_IN_RESIDENCE: Reduced to 3 different types
    valid_values_mr = ['< 1 year', '1-2 years', '+ 2 years'] # valid values for splitting into categories
    missing_mr = df['MONTHS_IN_RESIDENCE'].isnull() # identify null values

    def categorize_residence(months):
        if months <= 12: return valid_values_mr[0]
        elif 12 < months <= 24: return valid_values_mr[1]
        else: return valid_values_mr[2]

    # Apply the previous function
    df['MONTHS_IN_RESIDENCE'] = df['MONTHS_IN_RESIDENCE'].apply(categorize_residence)

    # Getting the data distribution, excluding null and nan values.
    distribution_mr = df[df['MONTHS_IN_RESIDENCE'].isin(valid_values_mr)]['MONTHS_IN_RESIDENCE'].value_counts(normalize=True)

    # Changing nan and null values with the distribution.
    df.loc[missing_mr, 'MONTHS_IN_RESIDENCE'] = np.random.choice(distribution_mr.index, size=len(df[missing_mr]), p=distribution_mr.values)
    df['MONTHS_IN_RESIDENCE'] = df['MONTHS_IN_RESIDENCE'].astype('category') # Set this column as category type


    # FLAG_EMAIL: has no null values, changed to category
    df['FLAG_EMAIL'] = df['FLAG_EMAIL'].astype('category')


    # 'PERSONAL_MONTHLY_INCOME_LEVEL': 5 different ranges (<500, 501-1000, 1001-1500, 1500-2001, + 2000), changed to category
    df['PERSONAL_MONTHLY_INCOME_LEVEL'] = np.where(df['PERSONAL_MONTHLY_INCOME'] < 501, '< 500',
                np.where(df['PERSONAL_MONTHLY_INCOME'] < 1001, '501 - 1000',
                np.where(df['PERSONAL_MONTHLY_INCOME'] < 1500, '1001 - 1500',
                np.where(df['PERSONAL_MONTHLY_INCOME'] < 2000, '1501 - 2000', '+ 2000'))))
    df['PERSONAL_MONTHLY_INCOME_LEVEL'] = df['PERSONAL_MONTHLY_INCOME_LEVEL'].astype('category') # Set this column as category type


    # 'OTHER_INCOMES' changed for 'OTHER_INCOMES' with 2 values, Y, N
    df['EXTRA_INCOME'] = ['Y' if income > 0 else 'N' for income in df['OTHER_INCOMES']]
    df['EXTRA_INCOME'] = df['EXTRA_INCOME'].astype('category') # Set this column as category type


    # 'FLAG_VISA', 'FLAG_MASTERCARD', 'FLAG_DINERS', 'FLAG_AMERICAN_EXPRESS', 'FLAG_OTHER_CARDS'
    # changed for 'FLAG_CARDS', has been set to 2 values Y, N, type category
    df['FLAG_CARDS'] = np.where(df['FLAG_VISA']>0, 'Y', 
                np.where(df['FLAG_MASTERCARD']>0, 'Y',
                np.where(df['FLAG_DINERS']>0, 'Y',
                np.where(df['FLAG_AMERICAN_EXPRESS']>0, 'Y',
                np.where(df['FLAG_OTHER_CARDS']>0, 'Y', 'N')))))
    df['FLAG_CARDS'] = df['FLAG_CARDS'].astype('category') # Changed to category


    # 'QUANT_BANKING_ACCOUNTS': changed for 'HAS_BANKING_ACCOUNTS' and set to 2 values, Y, N
    df['HAS_BANKING_ACCOUNTS'] = np.where(df['QUANT_BANKING_ACCOUNTS']>0, 'Y', 'N')
    df['HAS_BANKING_ACCOUNTS'] = df['HAS_BANKING_ACCOUNTS'].astype('category') # Changed to category


    # 'QUANT_SPECIAL_BANKING_ACCOUNTS': changed for 'HAS_SPECIAL_BANKING_ACCOUNTS', set to 2 values, Y, N
    df['HAS_SPECIAL_BANKING_ACCOUNTS'] = np.where(df['QUANT_SPECIAL_BANKING_ACCOUNTS']>0, 'Y', 'N')
    df['HAS_SPECIAL_BANKING_ACCOUNTS'] = df['HAS_SPECIAL_BANKING_ACCOUNTS'].astype('category') # Changed to category


    # 'PERSONAL_ASSETS_VALUE': changed for  set 2 values, Y, N
    df['HAS_PERSONAL_ASSETS'] = np.where(df['PERSONAL_ASSETS_VALUE']>0, 'Y', 'N')
    df['HAS_PERSONAL_ASSETS'] = df['HAS_PERSONAL_ASSETS'].astype('category') # Changed to category


    # 'QUANT_CARS': set 2 values, Y, N
    df['HAS_CARS'] = np.where(df['QUANT_CARS']>0, 'Y', 'N')
    df['HAS_CARS'] = df['QUANT_CARS'].astype('category') # Changed to category


    # 'OCCUPATION_TYPE' replace nan for weighted (numerical) values
    valid_values_ot = [1 , 2 , 3 , 4, 5, 0]  # the valid categories
    missing_ot = df['OCCUPATION_TYPE'].isnull() # Identifying the null and nan values
    distribution_ot = df[df['OCCUPATION_TYPE'].isin(valid_values_ot)]['OCCUPATION_TYPE'].value_counts(normalize=True)

    # Changing nan and null values with the distribution.
    df.loc[missing_ot, 'OCCUPATION_TYPE'] = np.random.choice(distribution_ot.index, size=len(df[missing_ot]), p=distribution_ot.values)
    df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].astype('category') # Changed to category


    # 'MATE_EDUCATION_LEVEL' replace nan for weighted values
    valid_values_ed = [1 , 2 , 3 , 4, 5, 0]  # Valid categories
    missing_ed = df['MATE_EDUCATION_LEVEL'].isnull() # Identifying the null and nan values
    distribution_ed = df[df['MATE_EDUCATION_LEVEL'].isin(valid_values_ed)]['MATE_EDUCATION_LEVEL'].value_counts(normalize=True)

    # Changing nan and null values with the distribution.
    df.loc[missing_ed, 'MATE_EDUCATION_LEVEL'] = np.random.choice(distribution_ed.index, size=len(df[missing_ed]), p=distribution_ed.values)
    df['MATE_EDUCATION_LEVEL'] = df['MATE_EDUCATION_LEVEL'].astype('category') # Changed to category


    # 'PRODUCT' changed to category, has no missing
    df['PRODUCT'] = df['PRODUCT'].astype('category')


    # 'AGE' changed to category, values every 10 years starting from 20 to + 60 years. 
    df['AGE'] = np.where(df['AGE'] < 21, '0 - 20 Years', 
            np.where(df['AGE'] < 31, '21 - 30 Years',
            np.where(df['AGE'] < 41, '31 - 40 Years',
            np.where(df['AGE'] < 51, '41 - 50 Years', '+ 60 Years'))))
    df['AGE'] = df['AGE'].astype('category') # Changed to category



    try:
        df.drop(columns=['PAYMENT_DAY', # replaced by PAYMENT_DAY_C
            'QUANT_DEPENDANTS', # replaced with HAS_DEPENDANTS
            'RESIDENCE_TYPE', # has too many empty rows
            'OTHER_INCOMES', # replaced with EXTRA_INCOME, 
            'FLAG_VISA', 'FLAG_MASTERCARD', 'FLAG_DINERS', # replaced with FLAG_CARDS
            'FLAG_AMERICAN_EXPRESS', 'FLAG_OTHER_CARDS', # replaced with FLAG_CARDS
            'QUANT_BANKING_ACCOUNTS', # replaced with HAS_BANKING_ACCOUNTS 
            'QUANT_SPECIAL_BANKING_ACCOUNTS', # replaced with HAS_SPECIAL_BANKING_ACCOUNTS
            'PERSONAL_ASSETS_VALUE', # replaced whit 'HAS_PERSONAL_ASSETS'
            'MONTHS_IN_THE_JOB', # 90% of values = 0
            'QUANT_CARS', # Was Changed for 'HAS_CARS'
            'PROFESSION_CODE', # Without relevant information abot this codes
            'MATE_PROFESSION_CODE', # Without relevant information abot this codes
            'PERSONAL_MONTHLY_INCOME' # Was changed for 'PERSONAL_MONTHLY_INCOMES_LEVEL'
            ], inplace=True)
    except:
        print('Note: There are no columns to delete, it seems they have already been deleted')
    
    return df