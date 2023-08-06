# Use the official Python image as the base image
FROM python:3.8-slim as base

# Set the working directory in the container
WORKDIR /src

# Copy the ml_model.py script into the container
#COPY ml_model.py /app/ml_model.py
#COPY logistic_regression.pk /app/logistic_regression.pk

# Copy all files and directories from the models directory into the container
COPY ./ /src

# Install the required dependencies for the model
RUN pip install --upgrade pip && pip install --no-cache-dir -r /src/requirements.txt

FROM base as build
# Run the model script when the container starts
ENTRYPOINT ["python3", "ml_model.py"]
