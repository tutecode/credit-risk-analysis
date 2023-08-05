# middleware.py
from fastapi import Request
from fastapi.responses import JSONResponse
from typing import Callable
from models.ml_model import predict_target

async def ml_model_middleware(request: Request, call_next: Callable):
    try:
        # Get the form data from the request
        form_data = await request.form()

        # Extract the necessary fields for prediction
        name = form_data.get("name")
        age = form_data.get("age")
        sex = int(form_data.get("sex"))
        # ... extract other fields similarly ...

        # Call the ml_model's predict_target function
        result, prob = predict_target(name, age, sex, ...)  # pass all the required fields

        # Store the result and probability in the request's state
        request.state.prediction_result = result
        request.state.prediction_prob = prob

        response = await call_next(request)

        return response
    except Exception as e:
        # Handle any errors gracefully and return an error response
        return JSONResponse(status_code=500, content={"error": "Error occurred during prediction"})

