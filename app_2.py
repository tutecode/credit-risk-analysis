# Flask
from flask import Flask, render_template, request
# Data manipulation
import pandas as pd
# Matrices manipulation
import numpy as np
# Script logging
import logging
# ML model
import joblib
# JSON manipulation
import json
# Utilities
import sys
import os

# Current directory
current_dir = os.path.dirname(__file__)

# Flask app
app = Flask(__name__, static_folder = 'static', template_folder = 'template')

# Logging
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

# Function
def ValuePredictor(data = pd.DataFrame):
	# Model name
	model_name = 'bin/xgboostModel.pkl' # Aca tengo q poner el pkl
	# Directory where the model is stored
	model_dir = os.path.join(current_dir, model_name)
	# Load the model
	loaded_model = joblib.load(open(model_dir, 'rb'))
	# Predict the data
	result = loaded_model.predict(data)
	
	return result[0]

# Home page
@app.route('/')
def home():
	return render_template('index.html')

# Prediction page
@app.route('/prediction', methods = ['POST'])
def predict():
	if request.method == 'POST':
		# Get the data from form
		# client_id = request.form['client_id']
		clerk_type = request.form['clerk_type'] # NO SE TOMO EN CUENTA
		payment_day = request.form['payment_day'] # NUMERICAL
		submition_type = request.form['submition_type'] # WEB o Carga
		quant_additional_cards = request.form['quant_additional_cards']
		postal_address_type = request.form['postal_address_type']
		sex = request.form['sex']
		marital_status = request.form['marital_status']
		quant_depend = request.form['quant_depend']
		edu_level = request.form['edu_level']
		birth_state = request.form['birth_state']
		birth_city = request.form['birth_city']
		nationality = request.form['nationality']
		residential_state = request.form['residential_state']
		residential_city = request.form['residential_city']
		residential_borough = request.form['residential_borough']
		res_phone = request.form['res_phone']
		res_phone_area = request.form['res_phone_area']
		res_type = request.form['res_type']
		phone = request.form['phone']
		email = request.form['email']
		per_month_income = request.form['per_month_income']
		other_income = request.form['other_income']
		visa = request.form['visa']
		master = request.form['master']
		diners = request.form['diners']
		am = request.form['am']
        other_cards = request.form['other_cards']
        num_bank_acc = request.form['num_bank_acc']
        num_special_bank_acc = request.form['num_special_bank_acc']
		num_cars = request.form['num_cars']
		company = request.form['company']
		prof_state = request.form['prof_state']
		prof_city = request.form['prof_city']
		prof_borough = request.form['prof_borough']
		prof_phone = request.form['prof_phone']
		prof_phone_area = request.form['prof_phone_area']
		job_month = request.form['job_month']
		prof_code = request.form['prof_code']
		occupation = request.form['occupation']
		mate_prof = request.form['mate_prof']
		mate_edu_level = request.form['mate_edu_level']
		rg = request.form['rg']
		cpf = request.form['cpf']
		income = request.form['income']
		product = request.form['product']
		acsp = request.form['acsp']
		age = request.form['age']
		res_zip_3 = request.form['res_zip_3']
		prof_zip_3 = request.form['prof_zip_3']
	    #flash(f"The client's ID is {data[0]}")
	
	

		# Load template of JSON file containing columns name
		# Schema name
		schema_name = 'dataset/columns_set.json'
		# Directory where the schema is stored
		schema_dir = os.path.join(current_dir, schema_name)
		with open(schema_dir, 'r') as f:
			cols =  json.loads(f.read())
		schema_cols = cols['data_columns']

		# Parse the categorical columns
		# Column of dependents
		try:
			col = ('Dependents_' + str(dependents))
			if col in schema_cols.keys():
				schema_cols[col] = 1
			else:
				pass
		except:
			pass
		# Column of property area
		try:
			col = ('Property_Area_' + str(property_area))
			if col in schema_cols.keys():
				schema_cols[col] = 1
			else:
				pass
		except:
			pass

		# Parse the numerical columns
		schema_cols['ApplicantIncome'] = applicant_income
		schema_cols['CoapplicantIncome'] = coapplicant_income
		schema_cols['LoanAmount'] = loan_amount
		schema_cols['Loan_Amount_Term'] = loan_term
		schema_cols['Gender_Male'] = gender
		schema_cols['Married_Yes'] = marital_status
		schema_cols['Education_Not Graduate'] = education
		schema_cols['Self_Employed_Yes'] = self_employed
		schema_cols['Credit_History_1.0'] = credit_history

		# Convert the JSON into data frame
		df = pd.DataFrame(
				data = {k: [v] for k, v in schema_cols.items()},
				dtype = float
			)

		# Create a prediction
		print(df.dtypes)
		result = ValuePredictor(data = df)

		# Determine the output
		if int(result) == 1:
			prediction = 'Dear Mr/Mrs/Ms {name}, your loan is approved!'.format(name = name)
		else:
			prediction = 'Sorry Mr/Mrs/Ms {name}, your loan is rejected!'.format(name = name)

		# Return the prediction
		return render_template('prediction.html', prediction = prediction)
	
	# Something error
	else:
		# Return error
		return render_template('error.html', prediction = prediction)

if __name__ == '__main__':
    app.run(debug = True)