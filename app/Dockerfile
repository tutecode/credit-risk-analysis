# Use the official Python image as the base image
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9 as base

# Set the working directory in the container
WORKDIR /app

# Copy the contents of the app directory to the container
COPY ./ /app

# Install the required dependencies for the app
RUN pip install --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt

FROM base as build
# Run the main FastAPI app when the container starts
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
