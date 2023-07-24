from helper_functions import utils_helpdata, utils_plot
import pandas as pd

def get_info_from_data(app_train,app_test,columns_description):
    

    # Data dimension
    print(
        "Data dimension: {} rows and {} columns".format(
            len(app_train), len(app_train.columns)
        )
    )

    # Data dimension
    print(
        "Data dimension: {} rows and {} columns".format(
            len(app_test), len(app_test.columns)
        )
    )


    ### Column Description data

    # - There are 2 columns with the same name "EDUCATION_LEVEL" 
    # - First is for general education level
    # - Second is for mate education level
    # - The second column aggregate with "MATE_EDUCATION_LEVEL"
    metadata = columns_description

    meta_cols = metadata["Var_Title"].to_list()
    meta_cols[43] = "MATE_EDUCATION_LEVEL"

    # Set the new column to the train_data and test_data
    app_train.columns = meta_cols
    app_test.columns = meta_cols[:-1]



    ## Exploratory Data Analysis (EDA)
    ### Training data
    #### Distribution target column
    utils_plot.plot_target_variable_distribution(app_train, "TARGET_LABEL_BAD=1")
    #### Distribution number of columns of each data type
    utils_plot.plot_data_type_counts(app_train)
    #### Distribution of uniques values for categorical columns
    utils_plot.plot_unique_value_counts(app_train, )
    #### Distribution percentage of missing data for each column
    utils_plot.plot_missing_data(app_train,)
    #### Handle missing values
    utils_helpdata.display_missing_values(app_train)

    #**Note**: Consideration to remove missing values is based on a business logic. 
    # The concept of *garbage in garbage out* applies. Without any relevant domain knowledges of loan problem, 
    # the interpolation will lead to the biased result.

    #Instead of dropping the missing values brutally, 
    # we try to inspect the relevant variables in the data 
    # in order to suggest the consideration for the next analysis

    #### Analyzing distribution of variables
    # - Show distribution of credit amounts
    # * Analyzing PERSONAL_MONTHLY_INCOME
    #   * it is the applicant's personal regular monthly income in Brazilian currency (R$)
    #   * it will be cast to (dollars$)
    income_colname = "PERSONAL_MONTHLY_INCOME"
    other_colname = "PROFESSIONAL_CITY"

    # order by income all dataframe
    df_order_by_income = app_train.sort_values(
        by=[income_colname], ascending=False, ignore_index=True
    )

    # select the first 30 incomes (most expensive)
    df_first_incomes = df_order_by_income.loc[:30, [income_colname, other_colname]]

    #barplot of first expensive incomes by professional country
    utils_plot.barplot(df_first_incomes,other_colname,income_colname)


    # mapping meta_colname to idx
    metadata_dic = {colname: idx for idx, colname in enumerate(meta_cols)}
    
    # show list of types of variables
    number_field_names = app_train.select_dtypes("number").columns.to_list()
    category_field_names = app_train.select_dtypes(exclude="number").columns.to_list()

   

    # get indices from numerical columns using metadata to index mapping
    idxs_number_cols = [
        metadata_dic[number_colname] for number_colname in number_field_names
    ]

    # showing metadata info
    metadata.iloc[idxs_number_cols, [0, 2]]

    #### Show metadata of number columns
    utils_helpdata.display_metadata(app_train,number_field_names,metadata,metadata_dic)
    
    