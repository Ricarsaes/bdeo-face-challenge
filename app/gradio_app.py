# app/gradio_app.py
import os
import traceback

import gradio as gr
import torch
from PIL import Image, ImageDraw

# --- Clean Imports (relative to this file's location in 'app/') ---
from app.config_api import (
    DEVICE,
    MODEL_STATE_DICT_PATH,
)
from app.models_pytorch.model_pytorch import FaceMultitaskModel
from app.utils import postprocess_predictions, preprocess_image

# --- 1. Load the PyTorch Model ---
# This code runs once when the module is first imported.
MODEL_GLOBAL = None

try:
    model_state_dict_file_path = os.path.join("app", MODEL_STATE_DICT_PATH)

    if os.path.exists(model_state_dict_file_path):
        MODEL_GLOBAL = FaceMultitaskModel()
        MODEL_GLOBAL.load_state_dict(
            torch.load(model_state_dict_file_path, map_location=torch.device(DEVICE))
        )
        MODEL_GLOBAL.to(DEVICE)
        MODEL_GLOBAL.eval()
    else:
        print(
            f"❌ ERROR: Model state_dict file not found at '{model_state_dict_file_path}'"
        )
except Exception as e:
    print(f"❌ An error occurred during model loading: {e}")
    traceback.print_exc()
    MODEL_GLOBAL = None


# --- 2. Define the Prediction Function ---
def predict_attributes(input_image: Image.Image):
    if MODEL_GLOBAL is None:
        raise gr.Error(
            "Model is not loaded. Please check the console for errors during startup."
        )
    if input_image is None:
        return None, None

    try:
        image_pil = input_image.convert("RGB")
        image_input = preprocess_image(image_pil).to(DEVICE)

        with torch.no_grad():
            pred_age, pred_gender, pred_eyes = MODEL_GLOBAL(image_input)

        predictions_tuple = (pred_age.cpu(), pred_gender.cpu(), pred_eyes.cpu())
        results_dict = postprocess_predictions(predictions_tuple)

        # Create annotated image
        annotated_image = image_pil.copy()
        draw = ImageDraw.Draw(annotated_image)

        # Get current image dimensions
        original_w, original_h = image_pil.size

        eye_coords_normalized = results_dict["eye_coordinates"].normalized

        # Scale normalized coordinates (0-1) directly to current image size
        lx_scaled = eye_coords_normalized[0] * original_w
        ly_scaled = eye_coords_normalized[1] * original_h
        rx_scaled = eye_coords_normalized[2] * original_w
        ry_scaled = eye_coords_normalized[3] * original_h

        # Draw circles for eyes
        radius = max(3, int(0.02 * min(original_w, original_h)))

        # Left eye (blue)
        draw.ellipse(
            (
                lx_scaled - radius,
                ly_scaled - radius,
                lx_scaled + radius,
                ly_scaled + radius,
            ),
            outline="blue",
            width=max(1, radius // 2),
        )

        # Right eye (red)
        draw.ellipse(
            (
                rx_scaled - radius,
                ry_scaled - radius,
                rx_scaled + radius,
                ry_scaled + radius,
            ),
            outline="red",
            width=max(1, radius // 2),
        )

        label_output = {
            "Age": f"{results_dict['age']:.1f} years",
            "Gender": f"{results_dict['gender'].capitalize()} ({results_dict['gender_confidence']:.2%})",
            "Eye Coords (Norm)": f"[lx={results_dict['eye_coordinates'].normalized[0]:.3f}, ly={results_dict['eye_coordinates'].normalized[1]:.3f}, rx={results_dict['eye_coordinates'].normalized[2]:.3f}, ry={results_dict['eye_coordinates'].normalized[3]:.3f}]",
            "Eye Coords (Scaled)": f"Left({lx_scaled:.1f}, {ly_scaled:.1f}), Right({rx_scaled:.1f}, {ry_scaled:.1f})",
        }

        return annotated_image, label_output

    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"An error occurred during prediction: {e}")


# --- 3. Define the Gradio Interface object ---
# This `demo` object is defined in the module's global scope, making it importable.
with gr.Blocks(theme=gr.themes.Soft(), title="Face Attribute Prediction") as demo:
    gr.Markdown(
        """
        # 👱 Face Attribute Prediction
        Upload an image of a face to predict **age**, **gender**, and **eye locations**.
        This interface uses the same PyTorch model and processing logic as the project's FastAPI endpoint.
        """
    )
    with gr.Row(variant="panel"):
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Face Image")
            predict_btn = gr.Button("Predict", variant="primary")
        with gr.Column(scale=1):
            annotated_image = gr.Image(
                label="Annotated Result", interactive=False, height=400
            )
            label_output = gr.JSON(label="Predictions")

    predict_btn.click(
        fn=predict_attributes,
        inputs=input_image,
        outputs=[annotated_image, label_output],
    )
    input_image.change(
        fn=predict_attributes,
        inputs=input_image,
        outputs=[annotated_image, label_output],
    )

# --- 4. Add a launch block for direct execution ---
if __name__ == "__main__":
    print("Running Gradio app directly for testing...")
    demo.launch()
