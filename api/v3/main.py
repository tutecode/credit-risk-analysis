# main.py
from fastapi import FastAPI, Form, HTTPException, Request

import pandas as pd
import logging
import joblib
import redis
import os
from helper_function import ml_model
import json
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Union
from fastapi.staticfiles import StaticFiles

# Current directory
current_dir = os.path.dirname(__file__)

app = FastAPI()

# Set up logging
log_filename = "app_log.log"
logging.basicConfig(
    level=logging.ERROR,
    filename=log_filename,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
file_handler = logging.FileHandler(filename=log_filename)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logging.getLogger().addHandler(file_handler)

# Initialize Redis client
redis_client = redis.StrictRedis(host="redis", port=6379, decode_responses=True)

# Home page
@app.get("/")
def home():
    return {"message": "Welcome to the Loan Prediction API!"}

# Load static directory
app.mount("/static", StaticFiles(directory="static"), name="static")
# Load Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Render the loan prediction form using Jinja2 template
@app.get("/index/", response_class=HTMLResponse)
async def get_loan_prediction_form(request: Request):
    # Render the template with the necessary context
    return templates.TemplateResponse("index.html", {"request": request})

# Esto va en el index si no funciona
# <form action="http://localhost:8000/prediction" method="post">


# Prediction page
@app.post("/prediction")
def predict(
    request: Request,
    name: str = Form(...),  # Matias
    age: str = Form(...),  # 26-35
    sex: int = Form(...),  # Male=1, Female=0
    marital_status: str = Form(...),  # other, single
    monthly_income_tot: str = Form(...),  # 1320-3323
    payment_day: int = Form(...),  # 0 = 1-14, 1 = 15-30
    residential_state: str = Form(...),  # AL
    months_in_residence: Union[str, None] = Form(None),  # (6-12, >12)
    product: str = Form(...),  # (2, 7)
    flag_company:  Union[str, None] = Form(None),  # Y=1
    flag_dependants: Union[str, None] = Form(None),  # Y=1
    quant_dependants: int = Form(...),  # (1, 2, 3)
    flag_residencial_phone: Union[str, None] = Form(None),  # Y=1
    flag_professional_phone: Union[str, None] = Form(None),  # Y=1
    flag_email: Union[str, None] = Form(None),  # Y=1
    flag_cards: Union[str, None] = Form(None),  # Y=1
    flag_residence: Union[str, None] = Form(None),  # Y=1
    flag_banking_accounts: Union[str, None] = Form(None),  # Y=1
    flag_personal_assets: Union[str, None] = Form(None),  # Y=1
    flag_cars: Union[str, None] = Form(None),  # Y=1
    submission_type_web: Union[str, None] = Form(None),  # Web=1
):
    # Load template of JSON file containing columns name
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
            schema_cols[col] = 0
    except:
        pass

    # MARITAL_STATUS_ (other, single)
    try:
        col = "MARITAL_STATUS_" + str(marital_status)
        if col in schema_cols.keys():
            schema_cols[col] = 1
        else:
            schema_cols[col] = 0
    except:
        pass

    # MONTHLY_INCOMES_TOT_ (1320-3323, 3323-8560, 650-1320, >8560)
    try:
        col = "MONTHLY_INCOMES_TOT_" + str(monthly_income_tot)
        if col in schema_cols.keys():
            schema_cols[col] = 1
        else:
            schema_cols[col] = 0
    except:
        pass

    # QUANT_DEPENDANTS_ (1, 2, 3)
    try:
        col = "QUANT_DEPENDANTS_" + str(quant_dependants)
        if col in schema_cols.keys():
            schema_cols[col] = 1
        else:
            schema_cols[col] = 0
    except:
        pass

    # MONTHS_IN_RESIDENCE_ (6-12, >12)
    try:
        col = "MONTHS_IN_RESIDENCE_" + str(months_in_residence)
        if col in schema_cols.keys():
            schema_cols[col] = 1
        else:
            schema_cols[col] = 0
    except:
        pass

    # PRODUCT_ (2, 7)
    try:
        col = "PRODUCT_" + str(product)
        if col in schema_cols.keys():
            schema_cols[col] = 1
        else:
            schema_cols[col] = 0
    except:
        pass

    # AGE_ (26-35, 36-45, 46-60, <18, >60)
    try:
        col = "AGE_" + str(age)
        if col in schema_cols.keys():
            schema_cols[col] = 1
        else:
            schema_cols[col] = 0
    except:
        pass
    
    flag_company = True if flag_company == "on" else False
    flag_dependants = True if flag_dependants == "on" else False
    flag_residencial_phone = True if flag_residencial_phone == "on" else False
    flag_professional_phone = True if flag_professional_phone == "on" else False
    flag_email = True if flag_email == "on" else False
    flag_cards = True if flag_cards == "on" else False
    flag_residence = True if flag_residence == "on" else False
    flag_banking_accounts = True if flag_banking_accounts == "on" else False
    flag_personal_assets = True if flag_personal_assets == "on" else False
    flag_cars = True if flag_cars == "on" else False
    submission_type_web = True if submission_type_web == "on" else False



    # Parse the Numerical columns (One column)
    schema_cols["PAYMENT_DAY_15-30"] = payment_day
    schema_cols["APPLICATION_SUBMISSION_TYPE_Web"] = submission_type_web
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

    result = ml_model.predict_target(df)

    # Determine the output message
    if int(result) == 1:
        prediction = 'Sorry Mr/Mrs/Ms {name}, your loan is rejected!'.format(name = name)
    else:
        prediction = 'Dear Mr/Mrs/Ms {name}, your loan is approved!'.format(name = name)

    context = {"request": request, "result": prediction}
    # Return the prediction{"request": request}
    return templates.TemplateResponse('prediction.html',  context)