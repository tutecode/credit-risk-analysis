# main.py
from fastapi import FastAPI, Form, HTTPException
import pandas as pd
import logging
import joblib
import redis
import os
from helper_function import preprocessing, ml_model
import json


# Current directory
current_dir = os.path.dirname(__file__)

app = FastAPI()
logging.basicConfig(level=logging.ERROR)

# Initialize Redis client
redis_client = redis.StrictRedis(host="redis", port=6379, decode_responses=True)


# Home page
@app.get("/")
def home():
    return {"message": "Welcome to the Loan Prediction API!"}


# Prediction page
@app.post("/prediction")
def predict(
    name: str = Form(...),  # Matias
    age: str = Form(...),  # 26-35
    sex: int = Form(...),  # Male=1, Female=0
    marital_status: str = Form(...),  # other, single
    monthly_income_tot: str = Form(...),  # 1320-3323
    payment_day: int = Form(...),  # 0 = 1-14, 1 = 15-30
    residential_state: str = Form(...),  # AL
    months_in_residence: str = Form(...),  # (6-12, >12)
    product: str = Form(...),  # (2, 7)
    flag_company: int = Form(...),  # Y=1
    flag_dependants: int = Form(...),  # Y=1
    quant_dependants: int = Form(...),  # (1, 2, 3)
    flag_residencial_phone: int = Form(...),  # Y=1
    flag_professional_phone: int = Form(...),  # Y=1
    flag_email: int = Form(...),  # Y=1
    flag_cards: int = Form(...),  # Y=1
    flag_residence: int = Form(...),  # Y=1
    flag_banking_accounts: int = Form(...),  # Y=1
    flag_personal_assets: int = Form(...),  # Y=1
    flag_cars: int = Form(...),  # Y=1
):
    # Load template of JSON file containing columns name
    # Schema name
    # schema_name = "data/columns_set.json"
    schema_name = "columns_set.json"
    # Directory where the schema is stored
    schema_dir = os.path.join(current_dir, schema_name)
    with open(schema_dir, "r") as f:
        cols = json.loads(f.read())
    schema_cols = cols["data_columns"]

    # Parse the Categorical columns (Greater than one column)
    # RESIDENCIAL_STATE_ (AL, ...)
    try:
        col = "RESIDENCIAL_STATE_" + str(residential_state)
        if col in schema_cols.keys():
            schema_cols[col] = 1
        else:
            pass
    except:
        pass

    # MARITAL_STATUS_ (other, single)
    try:
        col = "MARITAL_STATUS_" + str(marital_status)
        if col in schema_cols.keys():
            schema_cols[col] = 1
        else:
            pass
    except:
        pass

    # MONTHLY_INCOMES_TOT_ (1320-3323, 3323-8560, 650-1320, >8560)
    try:
        col = "MONTHLY_INCOMES_TOT_" + str(monthly_income_tot)
        if col in schema_cols.keys():
            schema_cols[col] = 1
        else:
            pass
    except:
        pass

    # QUANT_DEPENDANTS_ (1, 2, 3)
    try:
        col = "QUANT_DEPENDANTS_" + str(quant_dependants)
        if col in schema_cols.keys():
            schema_cols[col] = 1
        else:
            pass
    except:
        pass

    # MONTHS_IN_RESIDENCE_ (6-12, >12)
    try:
        col = "MONTHS_IN_RESIDENCE_" + str(months_in_residence)
        if col in schema_cols.keys():
            schema_cols[col] = 1
        else:
            pass
    except:
        pass

    # PRODUCT_ (2, 7)
    try:
        col = "PRODUCT_" + str(product)
        if col in schema_cols.keys():
            schema_cols[col] = 1
        else:
            pass
    except:
        pass

    # AGE_ (26-35, 36-45, 46-60, <18, >60)
    try:
        col = "AGE_" + str(age)
        if col in schema_cols.keys():
            schema_cols[col] = 1
        else:
            pass
    except:
        pass

    # Parse the Numerical columns (One column)
    schema_cols["PAYMENT_DAY_15-30"] = payment_day
    # schema_cols["APPLICATION_SUBMISSION_TYPE_Web"] = 1
    schema_cols["FLAG_RESIDENCIAL_PHONE_Y"] = flag_residencial_phone
    schema_cols["FLAG_PROFESSIONAL_PHONE_Y"] = flag_professional_phone
    schema_cols["COMPANY_Y"] = flag_company
    schema_cols["FLAG_EMAIL_1"] = flag_email
    schema_cols["SEX_M"] = sex
    schema_cols["HAS_DEPENDANTS_True"] = flag_dependants
    schema_cols["HAS_RESIDENCE_True"] = flag_residence
    schema_cols["HAS_CARDS_True"] = flag_cards
    schema_cols["HAS_BANKING_ACCOUNTS_True"] = flag_banking_accounts
    schema_cols["HAS_PERSONAL_ASSETS_True"] = flag_personal_assets
    schema_cols["HAS_CARS_True"] = flag_cars

    # Convert the JSON into data frame
    df = pd.DataFrame(data={k: [v] for k, v in schema_cols.items()}, dtype=float)

    # Convert the input data dictionary into a DataFrame
    # df = pd.DataFrame(input_data, index=[0])

    # Use the trained model to make a prediction
    # df_normalized = preprocessing.overall(df)
    # df_encoded = ml_model.encoding(df_normalized, True)
    prediction = ml_model.predict_target(df)

    # prediction = predict_loan_approval(df_predicted)
    # # Determine the output message
    if prediction == 1:
        output_message = f"Dear Mr/Mrs/Ms {name}, your loan is approved!"
    else:
        output_message = f"Sorry Mr/Mrs/Ms {name}, your loan is rejected!"

    return {"prediction": prediction, "message": output_message}
