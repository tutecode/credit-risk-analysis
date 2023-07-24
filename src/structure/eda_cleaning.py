### Cleaning Data
# cloning app_train
target_col = "TARGET_LABEL_BAD=1"
app_train_cop = app_train.copy()
app_test_cop = app_test.copy()
#### Working on Numerical Data
# Column: ID_CLIENT

# delete ID_CLIENT because it is only an index
curr_col_name = "ID_CLIENT"

app_train_cop.drop(columns=[curr_col_name], inplace=True)
app_test_cop.drop(columns=[curr_col_name], inplace=True)

number_field_names = app_train.select_dtypes(include="number").columns.to_list()
number_field_names.remove(curr_col_name)
#'PAYMENT_DAY' : split every 15 days
# {0: "1quincena", 1:"2quincena"}
curr_col_name = "PAYMENT_DAY"
tmp_col = curr_col_name + "tmp"
# before of cleaning
plotting_distribution(curr_col_name, app_train_cop, app_test_cop, target_col)

bins = [0, 15, 30]
numerical = [0, 1]
category = ["1quinc", "2quinc"]
# categorical changes
app_train_cop[tmp_col] = pd.cut(app_train_cop[curr_col_name], bins, labels=category)
app_test_cop[tmp_col] = pd.cut(app_test_cop[curr_col_name], bins, labels=category)

percents_by_target = get_percents_by_target(tmp_col, app_train_cop, target_col)
fig, axes = plt.subplots(1, 2, figsize=(8, 2))
plot_value_counts(
    fig, axes, tmp_col, app_train_cop, app_test_cop, percents_by_target, target_col
)
#'QUANT_ADDITIONAL_CARDS' tiene solo un valor (0), se puede eliminar.
# idea: colocar como binario si tiene o no tiene una tarjeta adicional
# se reemplazaran tanto en test con el valor mas frequente de toda
# la distribucion (train + test), valor_frecuente = 0 (sin tarjeta)
# sigue siendo variable numerica
curr_col_name = "QUANT_ADDITIONAL_CARDS"
tmp_col = curr_col_name + "tmp"
# before of cleaning
plotting_distribution(curr_col_name, app_train_cop, app_test_cop, target_col)

# categorical changes

# cast to binary numerical categories {0: have a card, 1: don't have a card}
app_train_cop[tmp_col] = np.where(app_train_cop[curr_col_name] > 0, "N", "Y")
app_test_cop[tmp_col] = np.where(app_test_cop[curr_col_name] > 0, "N", "Y")

# impute nans in testing with 0 (most frequent from train)
imp_const_zero = SimpleImputer(
    missing_values=np.nan, strategy="constant", fill_value="Y"
)
app_test_cop[tmp_col] = imp_const_zero.fit_transform(app_test_cop[[tmp_col]]).ravel()

percents_by_target = get_percents_by_target(tmp_col, app_train_cop, target_col)
fig, axes = plt.subplots(1, 2, figsize=(8, 2))
plot_value_counts(
    fig, axes, tmp_col, app_train_cop, app_test_cop, percents_by_target, target_col
)
#'POSTAL_ADDRESS_TYPE' tiene (1=49673, 2=327) (don't apply changes) (REMOVED)
# se puede eliminar por la disparidad de valores.
# keep values it could be deleted after of seeing feature importances
curr_col_name = "POSTAL_ADDRESS_TYPE"
# before of cleaning
plotting_distribution(curr_col_name, app_train_cop, app_test_cop, target_col)
percents_by_target = get_percents_by_target(curr_col_name, app_train_cop, target_col)
fig, axes = plt.subplots(1, 2, figsize=(8, 2))
plot_value_counts(
    fig,
    axes,
    curr_col_name,
    app_train_cop,
    app_test_cop,
    percents_by_target,
    target_col,
)
# view the relation between age and marital status
# in order to set what is every code of marital status
curr_col_name = "MARITAL_STATUS"
print("Relationship between Age and Marital Status")
app_train_cop[["AGE", curr_col_name]].groupby([curr_col_name]).median()
# conclude marital status by year
# it has 35 years old -> single (a single person has few years old)
# it has 42 years old -> married (a married person has more years old)
#### Marital Status Code in Brazil
- regards to this [standard](https://international.ipums.org/international-action/variables/MARST#codes_section)

|code|marital status|
|---|---------------|
|0| NIU (not in universe)|
|1|	Single/never married *|	
|2|	Married/in union *|
|3|	Separated/divorced/spouse absent|
|4|	Widowed|
#'MARITAL_STATUS' tiene números del 0 al 7,
# la mayoría están concentrados en (1=15286, 2=25967),
# los otros números podemos codificarlo como indeterminado
# para no eliminar esta columnas.
curr_col_name = "MARITAL_STATUS"
# {single:1, married: 2, rest:3}
tmp_col = curr_col_name + "tmp"
# before of cleaning
plotting_distribution(curr_col_name, app_train_cop, app_test_cop, target_col)

# categorical changes
app_train_cop[tmp_col] = np.where(
    app_train_cop[curr_col_name] == 1,
    "single",
    np.where(app_train_cop[curr_col_name] == 2, "married", "Other"),
)
app_test_cop[tmp_col] = np.where(
    app_test_cop[curr_col_name] == 1,
    "single",
    np.where(app_test_cop[curr_col_name] == 2, "married", "Other"),
)
percents_by_target = get_percents_by_target(tmp_col, app_train_cop, target_col)
fig, axes = plt.subplots(1, 2, figsize=(8, 2))
plot_value_counts(
    fig, axes, tmp_col, app_train_cop, app_test_cop, percents_by_target, target_col
)
# 'QUANT_DEPENDANTS' tiene números de 0 hasta el 13 y el 53,
# podemos clasificar con si y no, o en 0, 1, 2, 3 y 4+.
# summarizing, in range[0,1,2,3,4+]
# there exist a scale dependence between every numeric value
# so it keeps as a numerical value
curr_col_name = "QUANT_DEPENDANTS"
# before of cleaning
plotting_distribution(curr_col_name, app_train_cop, app_test_cop, target_col)
# numerrical changes
tmp_col = create_tmp_column(curr_col_name, app_train_cop, app_test_cop)
app_train_cop.loc[app_train_cop[curr_col_name] > 4, tmp_col] = 4
app_test_cop.loc[app_test_cop[curr_col_name] > 4, tmp_col] = 4
plotting_distribution(tmp_col, app_train_cop, app_test_cop, target_col)
#### Education System in Brazil coding

- start from Tertiary education

|   | Education | School/Level                           | Years |
|---|-----------|----------------------------------------|:-----:|
| - | Primary   | Ensino Fundamental (Elementary School) |   9   |
| 0 | Secondary | Ensino Médio (High School)             |   3   |
| 1 | Tertiary  | Higher Education- Ensino Superior      |       |
| 2 | Tertiary  | Bacharelado, Licenciado (Undergrad.)   |  4–6  |
| 3 | Tertiary  | Especialização (Graduate)              |   1   |
| 4 | Tertiary  | Mestre (Graduate)                      |  1–2  |
| 5 | Tertiary  | Doutor (Doctoral)                      |   2   |

[Education System in Brazil](https://www.scholaro.com/db/countries/brazil/education-system)
#'EDUCATION_LEVEL' solo tiene un numero 0, se puede eliminar. (REMOVED)
# 5 Profesional Education o 1
# 0:secundaria completa, 1:tienen estudios superiores
# there exist a scale dependence between every numeric value
# so it keeps as a numerical value
# the problem is match MATE EDUCATION LEVEL
# we don't know exactly the respective code, when predict
# and use an wrong code it could add bias to the result
# of the model so it will be remove
curr_col_name = "EDUCATION_LEVEL"
# before of cleaning
plotting_distribution(curr_col_name, app_train_cop, app_test_cop, target_col)
# 'NATIONALITY' los valores que tiene son (2=98, 0=2018, 1=47884) (REMOVED)
# se puede eliminar por la disparidad de valores.
# debe ser categorico y sera brazil:1 otros:2

curr_col_name = "NACIONALITY"  # correct translation -> NATIONALITY
tmp_col = curr_col_name + "tmp"
# before of cleaning
plotting_distribution(curr_col_name, app_train_cop, app_test_cop, target_col)

# categorical changes
app_train_cop[tmp_col] = np.where(app_train_cop[curr_col_name] == 1, "Brazil", "Other")
app_test_cop[tmp_col] = np.where(app_test_cop[curr_col_name] == 1, "Brazil", "Other")
percents_by_target = get_percents_by_target(tmp_col, app_train_cop, target_col)
fig, axes = plt.subplots(1, 2, figsize=(8, 2))
plot_value_counts(
    fig, axes, tmp_col, app_train_cop, app_test_cop, percents_by_target, target_col
)
|code|Residence Type|
|---|---|
|0|otros|
|1|owned|
|2|mortgage|
|3|rented|
|4|parents|
|5|family|
# 'RESIDENCE_TYPE' tiene valores del 0 hasta el 5,
# el de mayor proporcion es el 1 con 41572 filas,
# creo se puede eliminar al no tener mejor informacion.
# it should be categoric, value is maintained because of living zone

curr_col_name = "RESIDENCE_TYPE"
tmp_col = curr_col_name + "tmp"

# impute nans in testing with 0 (it already has a other category)
imp_const_zero = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)
app_train_cop[tmp_col] = imp_const_zero.fit_transform(
    app_train_cop[[curr_col_name]]
).ravel()
app_test_cop[tmp_col] = imp_const_zero.fit_transform(
    app_test_cop[[curr_col_name]]
).ravel()
# it doesn't need a numerical change only fill nans
# before of cleaning
plotting_distribution(tmp_col, app_train_cop, app_test_cop, target_col)

# categorical changes
app_train_cop[tmp_col] = np.where(
    app_train_cop[curr_col_name] == 1,
    "owned",
    np.where(
        app_train_cop[curr_col_name] == 2,
        "mortgage",
        np.where(
            app_train_cop[curr_col_name] == 3,
            "rented",
            np.where(
                app_train_cop[curr_col_name] == 4,
                "parents",
                np.where(app_train_cop[curr_col_name] == 4, "family", "other"),
            ),
        ),
    ),
)

app_test_cop[tmp_col] = np.where(
    app_test_cop[curr_col_name] == 1,
    "owned",
    np.where(
        app_test_cop[curr_col_name] == 2,
        "mortgage",
        np.where(
            app_test_cop[curr_col_name] == 3,
            "rented",
            np.where(
                app_test_cop[curr_col_name] == 4,
                "parents",
                np.where(app_test_cop[curr_col_name] == 4, "family", "other"),
            ),
        ),
    ),
)

percents_by_target = get_percents_by_target(tmp_col, app_train_cop, target_col)
fig, axes = plt.subplots(1, 2, figsize=(8, 2))
plot_value_counts(
    fig, axes, tmp_col, app_train_cop, app_test_cop, percents_by_target, target_col
)
- Regards to this Concepts [Normalization or Imputation](https://stats.stackexchange.com/questions/138203/imputation-of-missing-data-before-or-after-centering-and-scaling)

- First Normalization because of using standard values and working models imputers with small values,
other reason normalization avoid using bias from imputed values
only using bias from raw data with nans.
If you use imputing step first, it could add some bias, modifying distribution of values and getting other statistic when it applied normalization at the end.

- Regards to this Concepts [Scale with outliers](https://machinelearningmastery.com/robust-scaler-transforms-for-machine-learning/#:~:text=Standardization%20is%20calculated%20by%20subtracting,dividing%20by%20the%20standard%20deviation.&text=Sometimes%20an%20input%20variable%20may,are%20overrepresented%20for%20some%20reason.)
  
- Robust Scaler is a great standarization using interquartile range. It uses mean zero and standard deviation 1, and it could select interquartile range in order to consider more outliers with less window or few outliers with wide window
#'MONTH_IN_RESIDENCE' tiene cerda de 90 valores diferentes, y 3777 valores nulos,
# podemos normalizar por valores de cada 6 meses
# y los nulos se pueden cambiar por 0 a 6 meses.
# it should be numerical because of scalar dependency

# imputar nans con un valor estadistico,
# imputar zero representaria una persona que recien se ha mudado y pide un prestamo
# lo cual es poco probable
# al tener muchos valores de pequeña frecuencia
# el median podria crear una nueva categoria alejada de la distribucion
# el mean se acercara al medio de la distribucion
# por lo cual se imputara los nulls con un mean de datos
curr_col_name = "MONTHS_IN_RESIDENCE"
## Outliers detection
##### Helper functions to treats outliers
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
# look at relationship between target and months_in_residence
print("Relationship between Target and Months in Residence")
app_train_cop[["TARGET_LABEL_BAD=1", curr_col_name]].groupby([curr_col_name]).count()[
    ::6
]
#### Review this TODO

- Select in apropiately way the features which have a relevanc: [Feature Selection](https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b) 

1. Filter
2. Wrapper
3. Embedded
# Months_in_residence
# it must be numerical value for scalar dependency
curr_col_name = "MONTHS_IN_RESIDENCE"
tmp_col = curr_col_name + "tmp"
# before of cleaning
plotting_distribution(curr_col_name, app_train_cop, app_test_cop, target_col)


# 0:+3year, 1:3years, 2:2years, 3:1year
app_train_cop[tmp_col] = np.where(
    app_train_cop[curr_col_name] <= 1,
    3,
    np.where(
        app_train_cop[curr_col_name] <= 24,
        2,
        np.where(app_train_cop[curr_col_name] <= 36, 1, 0),
    ),
)
app_test_cop[tmp_col] = np.where(
    app_test_cop[curr_col_name] <= 1,
    3,
    np.where(
        app_test_cop[curr_col_name] <= 24,
        2,
        np.where(app_test_cop[curr_col_name] <= 36, 1, 0),
    ),
)

plotting_distribution(tmp_col, app_train_cop, app_test_cop, target_col)
# 'FLAG_EMAIL' tiene valores de 0=9886, 1=40114, (REMOVED)
# se puede usar como indicador,? aunque la diferencia es de 4 a 1.
curr_col_name = "FLAG_EMAIL"
# before of cleaning
plotting_distribution(curr_col_name, app_train_cop, app_test_cop, target_col)
#### Minimal Wage in Brazil
[Minimal Wage](https://www.remoti.io/blog/average-salary-in-brazil#:~:text=Minimum%20Wage,around%201.67%20USD%20per%20hour.). 
|Type of salary in Br|BRL/R$|
|--------------------|-----|
|minimum montlhy wage| 1320|
|average monthly cost of living| 3323|
|average monthly salary| 8560|
|highest-paid salary| 38200|

range of income achieve more than highest paid, it could be more than 40'000
# OTHER_INCOMES it could be add to personal income
# in order to increase people who has less than minimal salary
curr_col_name = "MONTHLY_INCOMES_TOT"
tmp_col = curr_col_name + "tmp"
app_train_cop[curr_col_name] = (
    app_train_cop["PERSONAL_MONTHLY_INCOME"] + app_train_cop["OTHER_INCOMES"]
)
app_test_cop[curr_col_name] = (
    app_test_cop["PERSONAL_MONTHLY_INCOME"] + app_test_cop["OTHER_INCOMES"]
)
# 'PERSONAL_MONTHLY_INCOME' no tiene valores nulos,
# se puede usar como indicador, pero hay que ajustar entre rangos,
# para mejorar la calidad de la columna.
# Separate by ranges but keeping numerical behaviour
# weight more to minimal ranges and less weight to great ranges
# focuse people who has minimal salary

# according to this
curr_col_name = "MONTHLY_INCOMES_TOT"
tmp_col = curr_col_name + "tmp"
# before of cleaning
plotting_distribution(curr_col_name, app_train_cop, app_test_cop, target_col)

# |minimum montlhy wage| [0-650]->5
# |minimum montlhy wage| [650-1320]->4
# |average monthly cost of living| [1320-3323]->3
# |average monthly salary| 8560| [3323-8560]->2
# |highest-paid salary| 38200| [>8560]->1

# it should be a numerical value

# Reglas:
app_train_cop[tmp_col] = np.where(
    app_train_cop[curr_col_name] < 650,
    "[0-650]",
    np.where(
        app_train_cop[curr_col_name] < 1320,
        "[650-1320]",
        np.where(
            app_train_cop[curr_col_name] < 3323,
            "[1320-3323]",
            np.where(app_train_cop[curr_col_name] < 8560, "[3323-8560]", "[>8560]"),
        ),
    ),
)

app_test_cop[tmp_col] = np.where(
    app_test_cop[curr_col_name] < 650,
    "[0-650]",
    np.where(
        app_test_cop[curr_col_name] < 1320,
        "[650-1320]",
        np.where(
            app_test_cop[curr_col_name] < 3323,
            "[1320-3323]",
            np.where(app_test_cop[curr_col_name] < 8560, "[3323-8560]", "[>8560]"),
        ),
    ),
)


percents_by_target = get_percents_by_target(tmp_col, app_train_cop, target_col)
fig, axes = plt.subplots(1, 2, figsize=(8, 2))
plot_value_counts(
    fig, axes, tmp_col, app_train_cop, app_test_cop, percents_by_target, target_col
)
# 'FLAG_VISA', 'FLAG_MASTERCARD', 'FLAG_DINERS',
# 'FLAG_AMERICAN_EXPRESS', 'FLAG_OTHER_CARDS'
# se podria colocar como si tiene o no tarjeta de credito, o cantidad de tarjetas de credito sin importar la marca.
# it was a good idea from Yessid to join flag cards
# and keep as a categorical variable
curr_col_name = "FLAG_CARDS"
tmp_col = curr_col_name

list_cards = [
    "FLAG_VISA",
    "FLAG_MASTERCARD",
    "FLAG_DINERS",
    "FLAG_AMERICAN_EXPRESS",
    "FLAG_OTHER_CARDS",
]


app_train_cop[tmp_col] = np.where(
    app_train_cop[list_cards[0]] > 0,
    "Y",
    np.where(
        app_train_cop[list_cards[1]] > 0,
        "Y",
        np.where(
            app_train_cop[list_cards[2]] > 0,
            "Y",
            np.where(
                app_train_cop[list_cards[3]] > 0,
                "Y",
                np.where(app_train_cop[list_cards[4]] > 0, "Y", "N"),
            ),
        ),
    ),
)

app_test_cop[tmp_col] = np.where(
    app_test_cop[list_cards[0]] > 0,
    "Y",
    np.where(
        app_test_cop[list_cards[1]] > 0,
        "Y",
        np.where(
            app_test_cop[list_cards[2]] > 0,
            "Y",
            np.where(
                app_test_cop[list_cards[3]] > 0,
                "Y",
                np.where(app_test_cop[list_cards[4]] > 0, "Y", "N"),
            ),
        ),
    ),
)

# after of cleaning
percents_by_target = get_percents_by_target(tmp_col, app_train_cop, target_col)
fig, axes = plt.subplots(1, 2, figsize=(8, 2))
plot_value_counts(
    fig, axes, tmp_col, app_train_cop, app_test_cop, percents_by_target, target_col
)
# 'QUANT_BANKING_ACCOUNTS'
# tiene 3 valores (2=14, 1=17864, 0=32122),
# se podria categorizar en si tiene o no tiene cuenta
# It should be a numerical value, quantity of cards
# could be influenced on target variable
curr_col_name = "QUANT_BANKING_ACCOUNTS"
tmp_col = "QUANT_BANKING_ACCOUNTS_TOT"

app_train_cop[tmp_col] = (
    app_train_cop["QUANT_BANKING_ACCOUNTS"]
    + app_train_cop["QUANT_SPECIAL_BANKING_ACCOUNTS"]
)
app_test_cop[tmp_col] = (
    app_test_cop["QUANT_BANKING_ACCOUNTS"]
    + app_test_cop["QUANT_SPECIAL_BANKING_ACCOUNTS"]
)
# look at relationship between target and months_in_residence
print("Relationship between Target and QUANT_BANKING_ACCOUNTS")
# people who has many banking cards is likely that they will pay the loan
# people who has few banking cards is likely that they won't pay the loan
app_train_cop[["TARGET_LABEL_BAD=1", "QUANT_BANKING_ACCOUNTS"]].groupby(
    ["QUANT_BANKING_ACCOUNTS"]
).count()
curr_col_name = "QUANT_BANKING_ACCOUNTS"
tmp_col = "QUANT_BANKING_ACCOUNTS_TOT"
# before of cleaning
plotting_distribution(curr_col_name, app_train_cop, app_test_cop, target_col)

# categorical changes
app_train_cop[tmp_col] = np.where(app_train_cop[curr_col_name] >= 2, "Y", "N")
app_test_cop[tmp_col] = np.where(app_test_cop[curr_col_name] >= 2, "Y", "N")

# after of cleaning
percents_by_target = get_percents_by_target(tmp_col, app_train_cop, target_col)
fig, axes = plt.subplots(1, 2, figsize=(8, 2))
plot_value_counts(
    fig, axes, tmp_col, app_train_cop, app_test_cop, percents_by_target, target_col
)
# PERSONAL ASSETS VALUE
# it achieves range from [0 to 6'000'000] of R$
# it could be a numerical value but with coded weight
curr_col_name = "PERSONAL_ASSETS_VALUE"
app_train_cop[["TARGET_LABEL_BAD=1", curr_col_name]].groupby([curr_col_name]).count()[
    ::10
]
# people who don't have any personal values likely pay their loans
curr_col_name = "PERSONAL_ASSETS_VALUE"
tmp_col = curr_col_name + "tmp"
# before of cleaning
plotting_distribution(curr_col_name, app_train_cop, app_test_cop, target_col)
# it has counting more in people who don't have any personal value
# so it could be cast in categorical as Y, N category
app_train_cop[tmp_col] = np.where(app_train_cop[curr_col_name] > 0, "Y", "N")
app_test_cop[tmp_col] = np.where(app_test_cop[curr_col_name] > 0, "Y", "N")
percents_by_target = get_percents_by_target(tmp_col, app_train_cop, target_col)
fig, axes = plt.subplots(1, 2, figsize=(8, 2))
plot_value_counts(
    fig, axes, tmp_col, app_train_cop, app_test_cop, percents_by_target, target_col
)
# QUANT_CARS' se podria cambiar si tiene carro o no,
# ya que los valores son 0 y 1.
# it should be a categorical value of Y,N
curr_col_name = "QUANT_CARS"
tmp_col = curr_col_name + "tmp"
app_train_cop[["TARGET_LABEL_BAD=1", curr_col_name]].groupby([curr_col_name]).count()
# an important personal active
# people who don't have a car likely won't pay the loan
# QUANT CARS
curr_col_name = "QUANT_CARS"
tmp_col = curr_col_name + "tmp"
# before of cleaning
plotting_distribution(curr_col_name, app_train_cop, app_test_cop, target_col)

app_train_cop[tmp_col] = np.where(app_train_cop[curr_col_name] == 1, "Y", "N")
app_test_cop[tmp_col] = np.where(app_test_cop[curr_col_name] == 1, "Y", "N")

percents_by_target = get_percents_by_target(tmp_col, app_train_cop, target_col)
fig, axes = plt.subplots(1, 2, figsize=(8, 2))
plot_value_counts(
    fig, axes, tmp_col, app_train_cop, app_test_cop, percents_by_target, target_col
)
# 'MONTHS_IN_THE_JOB' la mayoria de informacion se concentra
# range in [0-40] months 3.3years
# it should be a numerical  but
# it has more counts in 0 (people who probably don't have a job)
# so cast to categorical: Y (have a job),N(don't have a job)
curr_col_name = "MONTHS_IN_THE_JOB"
# print(app_train_cop[curr_col_name].value_counts().sort_values(ascending=False)[::2])
app_train_cop[["TARGET_LABEL_BAD=1", curr_col_name]].groupby([curr_col_name]).count()[
    ::2
]
# months in job (REMOVED)
curr_col_name = "MONTHS_IN_THE_JOB"
plotting_distribution(curr_col_name, app_train_cop, app_test_cop, target_col)
#### 'PROFESSION_CODE'
- searching codes only have a Brazil’s CBO94 codes but it has 
codes in group subgroups and general subgroups
- ranges of data is from [1-18],[1-27]

- If it has international code use [ISCO-08](https://www.ilo.org/public/english/bureau/stat/isco/docs/publication08.pdf), [ISCO-08 page](https://www.ilo.org/public/english/bureau/stat/isco/isco08/)

- If it has Brazil code use [CB094](http://www.mtecbo.gov.br/cbosite/pages/downloads.jsf), [CB094 page](http://www.mtecbo.gov.br/cbosite/pages/home.jsf)

- It is a good idea to delete this columne
- 1 reason it has many categories
- 2 reason it doesn't know the correct code of profession
- the high count is from code 9 but it could whatever profession in before standards, so it would be remove
- even if it has a good feature importance (TODO)
# 'PROFESSION_CODE' tiene mas de 7000 valores nulos, (REMOVED)
# hay valores del 0 hasta el 18,
# pienso que podemos eliminar la columna en vez de las celdas.
curr_col_name = "PROFESSION_CODE"
app_train_cop[curr_col_name].fillna(-1, inplace=True)
plotting_distribution(curr_col_name, app_train_cop, app_test_cop, target_col)
# remove_tmp_column(curr_col_name,app_train_cop,app_test_cop)
#### OCCUPATION TYPE
- in this case range is only 5 distinct values
- we could use the bih group of codes but again
- it is a guess, we don't know exactly the mapping of codes to occupation type.
- it is considered we could add bias if the user select
- a wrong code
- for this reason this column will be remove anyway it will checks the feature importance if we consider this column
#'OCCUPATION_TYPE' tiene valores del 0 hasta el 5, (REMOVED)
# y 7313 valores nulo, podriamos cambiar el nulo
# por desempleado (asumo que es valor de 0)
curr_col_name = "OCCUPATION_TYPE"
app_train_cop[curr_col_name].fillna(-1, inplace=True)
app_test_cop[curr_col_name].fillna(-1, inplace=True)
plotting_distribution(curr_col_name, app_train_cop, app_test_cop, target_col)
# remove_tmp_column(curr_col_name,app_train_cop,app_test_cop)
#'MATE_PROFESSION_CODE' tiene mas de 28 mil datos vacios, se puede eliminar.(REMOVED)
curr_col_name = "MATE_PROFESSION_CODE"
app_train_cop[curr_col_name].fillna(-1, inplace=True)
app_test_cop[curr_col_name].fillna(-1, inplace=True)
plotting_distribution(curr_col_name, app_train_cop, app_test_cop, target_col)
# remove_tmp_column(curr_col_name,app_train_cop,app_test_cop)
# MATE_EDUCATION_LEVEL (REMOVED)
# it could remove because consideably it has many missing
# values and we don't know exactly the codes of professions
# it will be analyzed with feature importance but
# it is considerably dangerous to guess the code in this
# category variable
curr_col_name = "MATE_EDUCATION_LEVEL"
app_train_cop[curr_col_name].fillna(-1, inplace=True)
app_test_cop[curr_col_name].fillna(-1, inplace=True)
plotting_distribution(curr_col_name, app_train_cop, app_test_cop, target_col)
# remove_tmp_column(curr_col_name,app_train_cop,app_test_cop)
# 'FLAG_HOME_ADDRESS_DOCUMENT' (REMOVED)
# solo tiene un valor de 0 para todas las filas,
# se puede eliminar.

# thinking on domain problem it will be replaced
# by RESIDENCE_TYPE and MONTHS_IN_RESIDENCE

# if we choose a flag with 2 values or vector of flags
# in RESIDENCE_TYPE or vector of number in MONTHS_IN_RESIDENCE
# it is better a vector

curr_col_name = "FLAG_HOME_ADDRESS_DOCUMENT"
plotting_distribution(curr_col_name, app_train_cop, app_test_cop, target_col)
# remove_tmp_column(curr_col_name,app_train_cop,app_test_cop)
# FLAG_RG (REMOVED)
# Flag indicating documental confirmation of citizen card number
# same as "FLAG_HOME_ADDRESS_DOCUMENT"
curr_col_name = "FLAG_RG"
plotting_distribution(curr_col_name, app_train_cop, app_test_cop, target_col)
# remove_tmp_column(curr_col_name,app_train_cop,app_test_cop)
# FLAG_CPF  (REMOVED)
# Flag indicating documental confirmation of tax payer status
# same as "FLAG_HOME_ADDRESS_DOCUMENT"
curr_col_name = "FLAG_CPF"
plotting_distribution(curr_col_name, app_train_cop, app_test_cop, target_col)
# remove_tmp_column(curr_col_name,app_train_cop,app_test_cop)
# FLAG_INCOME_PROOF  (REMOVED)
# Flag indicating documental confirmation of income
# same as "FLAG_HOME_ADDRESS_DOCUMENT"
curr_col_name = "FLAG_INCOME_PROOF"
plotting_distribution(curr_col_name, app_train_cop, app_test_cop, target_col)
# remove_tmp_column(curr_col_name,app_train_cop,app_test_cop)
# PRODUCT  (REMOVED)
# Type of credit product applied. Encoding not informed
# again it doesn't have exactly mapping codes
# it must be removed
curr_col_name = "PRODUCT"
plotting_distribution(curr_col_name, app_train_cop, app_test_cop, target_col)
# remove_tmp_column(curr_col_name,app_train_cop,app_test_cop)
# AGE (after of fi removed)
# Applicant's age at the moment of submission
# it should be a numerical value
curr_col_name = "AGE"
tmp_col = curr_col_name + "tmp"
proc_outliers(app_train_cop, curr_col_name)
proc_outliers(app_test_cop, curr_col_name)
plotting_distribution(curr_col_name, app_train_cop, app_test_cop, target_col)
# there exist people who have less than 18 years old
# but they can get a loan and people who has more than
# 70 years old and they can get a loan too

app_train_cop[["TARGET_LABEL_BAD=1", curr_col_name]].groupby([curr_col_name]).count()[
    -8:
]
#### Working with categorical columns
category_field_names = app_train.select_dtypes(exclude="number").columns.to_list()
for categorical_field in category_field_names:
    print(
        "{:<32}{:<8}{}".format(
            categorical_field,
            len(app_train[categorical_field].unique()),
            metadata.iloc[metadata_dic[categorical_field], 2],
        )
    )
for categorical_field in category_field_names:
    print(
        "{:<32}{:<8}{}".format(
            categorical_field,
            len(app_train[categorical_field].unique()),
            metadata.iloc[metadata_dic[categorical_field], 1],
        )
    )
curr_col_name = "APPLICATION_SUBMISSION_TYPE"
app_train_cop[["TARGET_LABEL_BAD=1", curr_col_name]].groupby([curr_col_name]).count()
# APPLICATION_SUBMISSION_TYPE
# 'APPLICATION_SUBMISSION_TYPE' tiene ('0' 'Carga' 'Web')
# 0 será dispuesto como Otro y tratado categorico
curr_col_name = "APPLICATION_SUBMISSION_TYPE"
app_train_cop.loc[app_train_cop[curr_col_name] == "0", curr_col_name] = "Other"
app_test_cop.loc[app_test_cop[curr_col_name] == "0", curr_col_name] = "Other"
percents_by_target = get_percents_by_target(curr_col_name, app_train_cop, target_col)
fig, axes = plt.subplots(1, 2, figsize=(8, 2))
plot_value_counts(
    fig,
    axes,
    curr_col_name,
    app_train_cop,
    app_test_cop,
    percents_by_target,
    target_col,
)
# 'SEX'
# se conserva y es asignada como categorica
# los valores N y " " serán asignadas como nueva categoria
# pero la cantidad de valores asociados es muy pequeño
# además en las prediciones no se cuenta con una categoria adicional
# por lo pronto se removerán dichas filas

curr_col_name = "SEX"
# only in train
app_train_cop.drop(
    app_train_cop[
        (app_train_cop[curr_col_name] == "N") | (app_train_cop[curr_col_name] == " ")
    ].index,
    inplace=True,
)
percents_by_target = get_percents_by_target(curr_col_name, app_train_cop, target_col)
fig, axes = plt.subplots(1, 2, figsize=(8, 2))
plot_value_counts(
    fig,
    axes,
    curr_col_name,
    app_train_cop,
    app_test_cop,
    percents_by_target,
    target_col,
)
curr_col_name = "RESIDENCIAL_STATE"
state_contperc = app_train_cop[curr_col_name].value_counts().values / len(app_train_cop)
state_names = np.array(app_train_cop[curr_col_name].value_counts().index)
states_others = [
    name for contperc, name in zip(state_contperc, state_names) if contperc < 0.002
]
states_others
# RESIDENCIAL_STATE
# it could be add some importance to the model
# because it is the current place where people live
# territory has a different economy so it has a good weight on model
curr_col_name = "RESIDENCIAL_STATE"
app_train_cop.loc[
    app_train_cop[curr_col_name].isin(states_others), curr_col_name
] = "Otros"
app_test_cop.loc[
    app_test_cop[curr_col_name].isin(states_others), curr_col_name
] = "Otros"
percents_by_target = get_percents_by_target(curr_col_name, app_train_cop, target_col)
fig, axes = plt.subplots(1, 2, figsize=(8, 8))
plot_value_counts(
    fig,
    axes,
    curr_col_name,
    app_train_cop,
    app_test_cop,
    percents_by_target,
    target_col,
)
# 'FLAG_PROFESSIONAL_PHONE'
# it could be a good disparity to set if a person has a job
# 'FLAG_RESIDENCIAL_PHONE'
# it could be a good disparity to set if a person has a house
curr_col_name = "FLAG_PROFESSIONAL_PHONE"
percents_by_target = get_percents_by_target(curr_col_name, app_train_cop, target_col)
fig, axes = plt.subplots(1, 2, figsize=(8, 2))
plot_value_counts(
    fig,
    axes,
    curr_col_name,
    app_train_cop,
    app_test_cop,
    percents_by_target,
    target_col,
)

curr_col_name = "FLAG_RESIDENCIAL_PHONE"
percents_by_target = get_percents_by_target(curr_col_name, app_train_cop, target_col)
fig, axes = plt.subplots(1, 2, figsize=(8, 2))
plot_value_counts(
    fig,
    axes,
    curr_col_name,
    app_train_cop,
    app_test_cop,
    percents_by_target,
    target_col,
)

curr_col_name = "FLAG_MOBILE_PHONE"
percents_by_target = get_percents_by_target(curr_col_name, app_train_cop, target_col)
fig, axes = plt.subplots(1, 2, figsize=(8, 2))
plot_value_counts(
    fig,
    axes,
    curr_col_name,
    app_train_cop,
    app_test_cop,
    percents_by_target,
    target_col,
)

curr_col_name = "COMPANY"
percents_by_target = get_percents_by_target(curr_col_name, app_train_cop, target_col)
fig, axes = plt.subplots(1, 2, figsize=(8, 2))
plot_value_counts(
    fig,
    axes,
    curr_col_name,
    app_train_cop,
    app_test_cop,
    percents_by_target,
    target_col,
)

# FLAG_ACSP_RECORD
# Flag indicating if the applicant
# has any previous credit delinquency
curr_col_name = "FLAG_ACSP_RECORD"
percents_by_target = get_percents_by_target(curr_col_name, app_train_cop, target_col)
fig, axes = plt.subplots(1, 2, figsize=(8, 2))
plot_value_counts(
    fig,
    axes,
    curr_col_name,
    app_train_cop,
    app_test_cop,
    percents_by_target,
    target_col,
)
# STATE_OF_BIRTH
# 'STATE_OF_BIRTH' tiene 29 valores diferentes incluyendo 
# un espacio vacio y XX,
# se eliminará dicha variable al no poseer mucha carga
# o aporte al modelo, ya que no hay mucha valor
# de importancia sobre la variable TARGET
domain_outside_columns = [
    "EDUCATION_LEVEL", # coding is not available
    "POSTAL_ADDRESS_TYPE", # few values in one category (disparity)
    "FLAG_EMAIL",
    "PERSONAL_MONTHLY_INCOME",
    "OTHER_INCOMES",
    "QUANT_BANKING_ACCOUNTS",
    "QUANT_SPECIAL_BANKING_ACCOUNTS",
    "FLAG_VISA",
    "FLAG_MASTERCARD",
    "FLAG_DINERS",
    "FLAG_AMERICAN_EXPRESS",
    "FLAG_OTHER_CARDS",
    "MONTHS_IN_THE_JOB",
    "PROFESSION_CODE",
    "OCCUPATION_TYPE",
    "MATE_PROFESSION_CODE",
    "MATE_EDUCATION_LEVEL",
    "FLAG_HOME_ADDRESS_DOCUMENT",
    "FLAG_RG",
    "FLAG_CPF",
    "FLAG_INCOME_PROOF",
    "PRODUCT",
    "AGE",
    "CLERK_TYPE",
    "STATE_OF_BIRTH",
    "CITY_OF_BIRTH",
    "RESIDENCIAL_CITY",
    "RESIDENCIAL_BOROUGH",
    "RESIDENCIAL_PHONE_AREA_CODE",
    "PROFESSIONAL_STATE",
    "PROFESSIONAL_CITY",
    "PROFESSIONAL_BOROUGH",
    "PROFESSIONAL_PHONE_AREA_CODE",
    "RESIDENCIAL_ZIP_3",
    "PROFESSIONAL_ZIP_3",
    "FLAG_ACSP_RECORD", # it has one only category in train
    "FLAG_MOBILE_PHONE", # it has one only category in train
    "QUANT_ADDITIONAL_CARDS", # it has one only category in train
    "QUANT_ADDITIONAL_CARDStmp", # it has one only category in train
    #aditional_columns
    "PAYMENT_DAY",
    "MARITAL_STATUS",
    "QUANT_DEPENDANTS",
    "RESIDENCE_TYPE",
    "MONTHS_IN_RESIDENCE",
    "PERSONAL_ASSETS_VALUE",
    "QUANT_CARS",
    "MONTHLY_INCOMES_TOT",
    "NACIONALITY"
]

list_not_find = []
list_removed = []
for outside_column in domain_outside_columns:
    if(outside_column in app_train_cop.columns):
        list_removed.append(outside_column)
        remove_tmp_column(outside_column,app_train_cop,app_test_cop)
    else:
        list_not_find.append(outside_column)

print("this columns were removed: ",list_removed)
print("this columns were not found: ",list_not_find)
app_train_cop.info()
app_train_cop.head()
colnames = app_train_cop.columns.to_list()
colnames_fixed = [colname[:-3] if colname[-3:]=="tmp" else colname for colname in colnames ]
app_train_cop.columns = colnames_fixed
colnames = app_test_cop.columns.to_list()
colnames_fixed = [colname[:-3] if colname[-3:]=="tmp" else colname for colname in colnames ]
app_test_cop.columns = colnames_fixed
app_train_cop.head()
app_test_cop.head()
domain_categories = [
    "PAYMENT_DAY",
    "MARITAL_STATUS",
    "RESIDENCE_TYPE",
    "MONTHLY_INCOMES_TOT",
    "FLAG_CARDS",
    "QUANT_BANKING_ACCOUNTS_TOT",
    "PERSONAL_ASSETS_VALUE",
    "QUANT_CARS",
    "APPLICATION_SUBMISSION_TYPE",
    "SEX",
    "RESIDENCIAL_STATE",
    "FLAG_PROFESSIONAL_PHONE",
    "FLAG_RESIDENCIAL_PHONE",
    "COMPANY",
    "NACIONALITY"
]
list_not_find = []
list_categorized = []
for column_to_cat in domain_categories:
    if(column_to_cat in app_train_cop.columns):
        print(column_to_cat)
        list_categorized.append(column_to_cat)
        cast_to_category(column_to_cat,app_train_cop,app_test_cop)
    else:
        list_not_find.append(column_to_cat)
print("this columns were categorized: ",list_categorized)
print("this columns were not found: ",list_not_find)
app_train_cop.head()
# show number of columns per data type
number_fields = len(app_train_cop.select_dtypes(include="number").columns)
object_fields = len(app_train_cop.select_dtypes(exclude="number").columns)
number_fields_o = len(app_train.select_dtypes(include="number").columns)
object_fields_o = len(app_train.select_dtypes(exclude="number").columns)

d = {
    "var_type": ["number", "object"],
    "quantity_cop": [number_fields, object_fields],
    "quantity_ori": [number_fields_o, object_fields_o],
}

quant_kind_vars = pd.DataFrame(data=d, index=[1, 2])

fig, axes = plt.subplots(1, 2, figsize=(8, 2), sharey=True)
fig.suptitle("Type columns Quantity Before and After Cleaning")
fig.align_labels()
sns.barplot(
    ax=axes[0], y=quant_kind_vars["var_type"], x=quant_kind_vars["quantity_ori"]
)
sns.barplot(
    ax=axes[1], y=quant_kind_vars["var_type"], x=quant_kind_vars["quantity_cop"]
)

axes[0].set_title("Number of Features")
axes[0].set_xlabel("Original Dataset")
axes[1].set_xlabel("New Dataset")

target_dist = plot.compute_stats_count(quant_kind_vars, "quantity_cop")
percents_cop = [i[2] for i in target_dist]
plot.percented_patches_second(axes[1], percents_cop)

target_dist = plot.compute_stats_count(quant_kind_vars, "quantity_ori")
percents_ori = [i[2] for i in target_dist]
plot.percented_patches_second(axes[0], percents_ori)

plt.tight_layout()

plt.show()
app_train_cop.head()
app_test_cop.head()
# order target at the final of dataset
current_cols_train = app_train_cop.columns.to_list()
idx_target = app_train_cop.columns.to_list().index("TARGET_LABEL_BAD=1")
if(app_train_cop.iloc[:,-1:].columns[0] != app_train_cop.iloc[:,idx_target:].columns[0]):
    features_cols = current_cols_train[:idx_target] + current_cols_train[idx_target+1:] + [current_cols_train[idx_target]]
    #crear un nuevo df
    app_train_cop = app_train_cop[features_cols]
else:
    print("Target is the last column")
app_train_cop.head()
app_train_cop.columns