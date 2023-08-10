import json
import logging
import os
import time
import uuid
from datetime import timedelta
from typing import Union

import database
import pandas as pd
import redis
import settings
import utils
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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

# Current directory
current_dir = os.path.dirname(__file__)

# Connect to Redis
db = redis.Redis(
    host=settings.REDIS_IP, port=settings.REDIS_PORT, db=settings.REDIS_DB_ID
)

# Your FastAPI app
app = FastAPI()

# Load static directory
app.mount("/static", StaticFiles(directory="static"), name="static")
# Load Jinja2 templates
templates = Jinja2Templates(directory="templates")


# Home page
@app.get("/")
def home():
    return {"message": "Welcome to the Loan Prediction API!"}


@app.get("/login")
async def login_page(request: Request):
    """
    Display the login page template.

    Parameters:
    - request (Request): FastAPI request object.

    Returns:
    - TemplateResponse: The rendered login page template with the request context.
    """
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/token", response_class=HTMLResponse)
async def login_for_access_token(
    request: Request, username: str = Form(...), password: str = Form(...)
):
    """
    Authenticate user and generate access token.

    Parameters:
    - request (Request): FastAPI request object.
    - username (str): User's username.
    - password (str): User's password.

    Returns:
    - HTMLResponse: Login page with error message if login fails, otherwise index.html content.
    """

    user = utils.authenticate_user(database.fake_users_db, username, password)

    if not user:
        error_message = "Incorrect username or password"
        # Return the login page with an error message
        return templates.TemplateResponse(
            "login.html", {"request": request, "error_message": error_message}
        )

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = utils.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )

    # Render the index.html page directly in the response
    index_page_content = templates.get_template("index.html").render(request=request)
    response = HTMLResponse(content=index_page_content)

    # Set the token as a cookie
    response.set_cookie("access_token", access_token)
    return response


# Render the loan prediction form using Jinja2 template
@app.get("/index/", response_class=HTMLResponse)
async def get_loan_prediction_form(request: Request):
    # Render the template with the necessary context
    return templates.TemplateResponse("index.html", {"request": request})


# Prediction page
@app.post("/prediction")
def predict(
    request: Request,
    name: str = Form(...),
    age: str = Form(...),
    sex: int = Form(...),
    marital_status: str = Form(...),
    monthly_income_tot: str = Form(...),
    payment_day: int = Form(...),
    residential_state: str = Form(...),
    months_in_residence: Union[str, None] = Form(None),
    product: str = Form(...),
    flag_company: Union[str, None] = Form(None),
    flag_dependants: Union[str, None] = Form(None),
    quant_dependants: int = Form(...),
    flag_residencial_phone: Union[str, None] = Form(None),
    flag_professional_phone: Union[str, None] = Form(None),
    flag_email: Union[str, None] = Form(None),
    flag_cards: Union[str, None] = Form(None),
    flag_residence: Union[str, None] = Form(None),
    flag_banking_accounts: Union[str, None] = Form(None),
    flag_personal_assets: Union[str, None] = Form(None),
    flag_cars: Union[str, None] = Form(None),
):
    """
    Handle user's credit prediction request using a machine learning model.

    Parameters:
    - request (Request): FastAPI request object.
    - name (str): User's name.
    - age (str): Age range.
    - sex (int): Gender (Male=1, Female=0).
    - marital_status (str): Marital status (other, single).
    - monthly_income_tot (str): Monthly income range.
    - payment_day (int): Payment day (0 = 1-14, 1 = 15-30).
    - residential_state (str): State of residence.
    - months_in_residence (Union[str, None]): Months in residence.
    - product (str): Product type.
    - flag_company (Union[str, None]): Flag for company.
    - flag_dependants (Union[str, None]): Flag for dependants.
    - quant_dependants (int): Number of dependants.
    - flag_residencial_phone (Union[str, None]): Flag for residential phone.
    - flag_professional_phone (Union[str, None]): Flag for professional phone.
    - flag_email (Union[str, None]): Flag for email.
    - flag_cards (Union[str, None]): Flag for cards.
    - flag_residence (Union[str, None]): Flag for residence.
    - flag_banking_accounts (Union[str, None]): Flag for banking accounts.
    - flag_personal_assets (Union[str, None]): Flag for personal assets.
    - flag_cars (Union[str, None]): Flag for cars.

    Returns:
    - TemplateResponse: FastAPI template response containing prediction outcome.
    """

    # Load template of JSON file containing columns name
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

    # Parse the Numerical columns (One column)
    schema_cols["PAYMENT_DAY_15-30"] = payment_day
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
    # Replace NaN values with 0.0
    df = df.fillna(0.0)

    # Generate an id for the classification then
    data_message = {"id": str(uuid.uuid4()), "data": df.iloc[0].values.tolist()}

    job_data = json.dumps(data_message)
    job_id = data_message["id"]

    # Send the job to the model service using Redis
    db.lpush(settings.REDIS_QUEUE, job_data)

    # Wait for result model
    # Loop until we received the response from our ML model
    while True:
        # Attempt to get model predictions using job_id
        output = db.get(job_id)

        if output is not None:
            # Process the result and extract prediction and score
            output = json.loads(output.decode("utf-8"))
            model_name = output["model_name"]
            model_score = output["model_score"]
            prediction = output["prediction"]
            score = output["score"]

            db.delete(job_id)
            break

        # Sleep some time waiting for model results
        time.sleep(settings.API_SLEEP)

    # Determine the output message
    if int(prediction) == 1:
        prediction = "Sorry Mr/Mrs/Ms {name}, your loan is rejected!\n with a probability of {proba}".format(
            name=name, proba=score
        )
    else:
        prediction = "Dear Mr/Mrs/Ms {name}, your loan is approved!".format(name=name)

    context = {
        "request": request,
        "model_name": model_name,
        "model_score": model_score,
        "result": prediction,
    }

    # Render the prediction response using Jinja2 templates and return it
    return templates.TemplateResponse("prediction.html", context)
