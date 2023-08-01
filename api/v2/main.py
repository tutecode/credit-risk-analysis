# main.py
from fastapi import FastAPI, Form, HTTPException
import pandas as pd
import logging
import joblib
import redis
import os

app = FastAPI()
logging.basicConfig(level=logging.ERROR)

# Function to predict loan approval
def predict_loan_approval(input_data):
    # Replace this with your actual machine learning model prediction code
    # For demonstration purposes, we'll return a random prediction (0 or 1)
    import random
    return random.randint(0, 1)

# Initialize Redis client
redis_client = redis.StrictRedis(host='redis', port=6379, decode_responses=True)

# Home page
@app.get("/")
def home():
    return {"message": "Welcome to the Loan Prediction API!"}

# Prediction page
@app.post("/prediction")
def predict(name: str = Form(...),
            clerk_type: str = Form(...),
            payment_day: int = Form(...),
            application_submission_type: str = Form(...),
            quant_additional_cards: int = Form(...),
            postal_address_type: int = Form(...),
            sex: str = Form(...),
            marital_status: int = Form(...),
            quant_dependants: int = Form(...),
            education_level: int = Form(...),
            state_of_birth: str = Form(...),
            city_of_birth: str = Form(...),
            nacionality: int = Form(...),
            residential_state: str = Form(...),
            residential_city: str = Form(...),
            residential_borough: str = Form(...),
            flag_residencial_phone: str = Form(...),
            residential_phone_area_code: str = Form(...),
            residence_type: float = Form(...),
            months_in_residence: float = Form(...),
            flag_mobile_phone: str = Form(...),
            flag_email: int = Form(...),
            personal_monthly_income: float = Form(...),
            other_incomes: float = Form(...),
            flag_visa: int = Form(...),
            flag_mastercard: int = Form(...),
            flag_diners: int = Form(...),
            flag_american_express: int = Form(...),
            flag_other_cards: int = Form(...),
            quant_banking_accounts: int = Form(...),
            quant_special_banking_accounts: int = Form(...),
            personal_assets_value: float = Form(...),
            quant_cars: int = Form(...),
            company: str = Form(...),
            professional_state: str = Form(...),
            professional_city: str = Form(...),
            professional_borough: str = Form(...),
            flag_professional_phone: str = Form(...),
            professional_phone_area_code: str = Form(...),
            months_in_the_job: int = Form(...),
            profession_code: float = Form(...),
            occupation_type: float = Form(...),
            mate_profession_code: float = Form(...),
            home_address_document: int = Form(...),
            flag_rg: int = Form(...),
            flag_cpf: int = Form(...),
            flag_income_proof: int = Form(...),
            product: int = Form(...),
            flag_acsp_record: str = Form(...),
            age: int = Form(...),
            residencial_zip_3: str = Form(...),
            professional_zip_3: str = Form(...),
           ):

    # Combine form fields into a dictionary to represent the input data
    input_data = {
        #"name": name, NOT NAME
        "clerk_type": clerk_type,
        "payment_day": payment_day,
        "application_submission_type": application_submission_type,
        "quant_additional_cards": quant_additional_cards,
        "postal_address_type": postal_address_type,
        "sex": sex,
        "marital_status": marital_status,
        "quant_dependants": quant_dependants,
        "education_level": education_level,
        "state_of_birth": state_of_birth,
        "city_of_birth": city_of_birth,
        "nacionality": nacionality,
        "residential_state": residential_state,
        "residential_city": residential_city,
        "residential_borough": residential_borough,
        "flag_residencial_phone": flag_residencial_phone,
        "residential_phone_area_code": residential_phone_area_code,
        "residence_type": residence_type,
        "months_in_residence": months_in_residence,
        "flag_mobile_phone": flag_mobile_phone,
        "flag_email": flag_email,
        "personal_monthly_income": personal_monthly_income,
        "other_incomes": other_incomes,
        "flag_visa": flag_visa,
        "flag_mastercard": flag_mastercard,
        "flag_diners": flag_diners,
        "flag_american_express": flag_american_express,
        "flag_other_cards": flag_other_cards,
        "quant_banking_accounts": quant_banking_accounts,
        "quant_special_banking_accounts": quant_special_banking_accounts,
        "personal_assets_value": personal_assets_value,
        "quant_cars": quant_cars,
        "company": company,
        "professional_state": professional_state,
        "professional_city": professional_city,
        "professional_borough": professional_borough,
        "flag_professional_phone": flag_professional_phone,
        "professional_phone_area_code": professional_phone_area_code,
        "months_in_the_job": months_in_the_job,
        "profession_code": profession_code,
        "occupation_type": occupation_type,
        "mate_profession_code": mate_profession_code,
        "home_address_document": home_address_document,
        "flag_rg": flag_rg,
        "flag_cpf": flag_cpf,
        "flag_income_proof": flag_income_proof,
        "product": product,
        "flag_acsp_record": flag_acsp_record,
        "age": age,
        "residencial_zip_3": residencial_zip_3,
        "professional_zip_3": professional_zip_3
    }

    # Convert the input data dictionary into a DataFrame
    df = pd.DataFrame(input_data, index=[0])

    # Apply any data preprocessing, if required

    # Use the trained model to make a prediction
    prediction = predict_loan_approval(df)

    # Determine the output message
    if prediction == 1:
        output_message = f"Dear Mr/Mrs/Ms {name}, your loan is approved!"
    else:
        output_message = f"Sorry Mr/Mrs/Ms {name}, your loan is rejected!"

    return {"prediction": prediction, "message": output_message}
