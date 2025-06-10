import importlib.util
import io
import os

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from app.config_api import DEVICE, MODEL_CLASS_NAME, MODEL_MODULE, MODEL_STATE_DICT_PATH

# Import Pydantic models
from app.schemas import ErrorResponse, PredictionResponse
from app.utils import postprocess_predictions, preprocess_image

import traceback

app = FastAPI(
    title="Face Attribute Prediction API",
    description="API to predict age, gender, and eye coordinates from face images",
    version="1.0.0",
)


MODEL_GLOBAL = None
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))

    model_definition_file_path = os.path.join(
        current_dir, "models_pytorch", "model_pytorch.py"
    )

    model_state_dict_file_path = os.path.join(current_dir, MODEL_STATE_DICT_PATH)

    if os.path.exists(model_definition_file_path) and os.path.exists(
        model_state_dict_file_path
    ):

        spec = importlib.util.spec_from_file_location(
            MODEL_MODULE, model_definition_file_path
        )
        ModelClassModule = importlib.util.module_from_spec(spec)

        spec.loader.exec_module(ModelClassModule)
        ModelClass = getattr(ModelClassModule, MODEL_CLASS_NAME)

        MODEL_GLOBAL = ModelClass()
        MODEL_GLOBAL.load_state_dict(
            torch.load(model_state_dict_file_path, map_location=torch.device(DEVICE))
        )
        MODEL_GLOBAL.to(DEVICE)
        MODEL_GLOBAL.eval()
        print(
            f"Model loaded successfully from {model_state_dict_file_path} and moved to {DEVICE}."
        )
    else:
        if not os.path.exists(model_definition_file_path):
            print(f"Model definition file not found: {model_definition_file_path}")
        if not os.path.exists(model_state_dict_file_path):
            print(f"Model state_dict file not found: {model_state_dict_file_path}")
        MODEL_GLOBAL = None
except Exception as e:
    print(f"Error loading  model: {e}")
    MODEL_GLOBAL = None


common_responses = {
    503: {
        "model": ErrorResponse,
        "description": "Model is not available or failed to load.",
    },
    500: {
        "model": ErrorResponse,
        "description": "Internal server error during prediction.",
    },
}


@app.post(
    "/predict/pytorch", response_model=PredictionResponse, responses=common_responses
)
async def predict_pytorch_endpoint(
    file: UploadFile = File(
        ..., description="Image file for face attribute prediction."
    )
):
    if MODEL_GLOBAL is None:
        raise HTTPException(
            status_code=503, detail="Model is not available or failed to load."
        )
    try:
        image_bytes = await file.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_input = preprocess_image(image_pil).to(DEVICE)

        with torch.no_grad():
            pred_age, pred_gender, pred_eyes = MODEL_GLOBAL(image_input)

        predictions = (pred_age.cpu(), pred_gender.cpu(), pred_eyes.cpu())
        results_dict = postprocess_predictions(predictions)
        results_dict["model_type"] = "pytorch"
        return PredictionResponse(**results_dict)
    except Exception as e:
        print(f"Prediction error: {str(e)}")

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Error during prediction: {str(e)}"
        )


@app.get(
    "/",
    summary="API Root",
    description="Welcome message for the Face Attribute Prediction API.",
)
async def root():
    return {
        "message": "Face Attribute Prediction API. Access /docs for API documentation and endpoints."
    }


# To run locally for testing before Docker:
# From face_attribute_project/ directory:
# python -m uvicorn app.main:app --reload --port 8000
