# Use the official Python image as the base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the ml_model.py script into the container
#COPY ml_model.py /app/ml_model.py
#COPY logistic_regression.pk /app/logistic_regression.pk

# Copy all files and directories from the models directory into the container
COPY . /app

# Install the required dependencies for the model
RUN pip install --no-cache-dir -r /app/requirements.txt

# Run the model script when the container starts
CMD ["python", "ml_model.py"]
