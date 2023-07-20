from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# repeated column
def repeated_column(df):
    dupla = df.columns[df.columns.duplicated()].to_list()
    location = [i for i, j in enumerate(df) if j==dupla[0]]

    print(f'The repeated column is {dupla}, its index are {location}')

# rename the repeated column
def rename_column(df1, df2):
    meta_cols = df2["Var_Title"].to_list()
    meta_cols[43] = "MATE_EDUCATION_LEVEL"
    df1.columns = meta_cols

# check missing data / show columns and number of missing data
def missing_data(df):
    index_missings = df.isna().sum().index
    missing_count = df.isna().sum()
    missing_perc = df.isna().sum()/len(df)*100

    ind_missing_count = list(zip(index_missings,missing_count,missing_perc))
    ind_missing_count.sort(key=lambda x:x[1],reverse=True)

    print("{:<28} {:<8} {:<5}".format("","Total","Percent"))
    for missing in ind_missing_count[:20]:
        print("{:<28} {:<8} {:<5.1f}".format(missing[0], missing[1], missing[2]))

# number columns for data type / show number of columns per data type
def number_data(df):
    float_fields = len(df.select_dtypes("float64").columns)
    int_fields = len(df.select_dtypes("int64").columns)
    object_fields = len(df.select_dtypes("object").columns)
    category_fields = len(df.select_dtypes("category").columns)

    print(f"float64: {float_fields}")
    print(f"int64: {int_fields}")
    print(f"object: {object_fields}")
    print(f"category: {category_fields}")


# number of unique values / only works for object and categorycal type
def unique_values(df, string):
    # show number of unique values per categorical column
    object_field_name = df.select_dtypes(string).columns.to_list()
    for object_field in object_field_name:
        print("{:<30}{}".format(object_field,len(df[object_field].unique())))


# this function creates a column with their importance order vs target column
def col_vs_target(df):

    # The target column is assigned to 'y' and the characteristics is ssigned to 'X'
    y = df['TARGET_LABEL_BAD=1']
    X = df.drop(columns='TARGET_LABEL_BAD=1')

    # Imputing the lost data (is used only for showing purpose)
    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = X[col].astype(str)
            X[col] = cat_imputer.fit_transform(X[col].values.reshape(-1, 1))
        else:
            X[col] = num_imputer.fit_transform(X[col].values.reshape(-1, 1))

    # Code the categoical 
    label_encoder = LabelEncoder()
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = label_encoder.fit_transform(X[col])

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ajustar el modelo
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Obtener la importancia de las características
    importance = pd.DataFrame(data={
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print(importance)
