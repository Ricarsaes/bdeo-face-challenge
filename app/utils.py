# app/utils.py
import torch
from PIL import Image
from torchvision import transforms

from app.config_api import (
    IMG_HEIGHT,
    IMG_WIDTH,
    MAX_AGE,
    ORIG_IMG_HEIGHT,
    ORIG_IMG_WIDTH,
)
from app.schemas import EyeCoordinates

pytorch_image_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
pytorch_preprocess_transform = transforms.Compose(
    [
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        pytorch_image_normalize,
    ]
)


def preprocess_image(image_pil: Image.Image) -> torch.Tensor:
    image_tensor = pytorch_preprocess_transform(image_pil)
    return image_tensor.unsqueeze(0)


def postprocess_predictions(predictions_pytorch) -> dict:
    pred_ages_norm_tensor, pred_genders_logits_tensor, pred_eyes_norm_tensor = (
        predictions_pytorch
    )

    age_actual = float(pred_ages_norm_tensor.item() * MAX_AGE)
    gender_prob = float(torch.sigmoid(pred_genders_logits_tensor).item())
    gender_pred_label = "female" if gender_prob > 0.5 else "male"

    eye_coords_normalized_list = [
        float(coord) for coord in pred_eyes_norm_tensor.squeeze().tolist()
    ]
    eye_coords_pixel_orig_ref_list = [
        round(eye_coords_normalized_list[0] * ORIG_IMG_WIDTH, 1),
        round(eye_coords_normalized_list[1] * ORIG_IMG_HEIGHT, 1),
        round(eye_coords_normalized_list[2] * ORIG_IMG_WIDTH, 1),
        round(eye_coords_normalized_list[3] * ORIG_IMG_HEIGHT, 1),
    ]

    return {
        "age": round(age_actual, 1),
        "gender": gender_pred_label,
        "gender_confidence": round(gender_prob, 4),
        "eye_coordinates": EyeCoordinates(
            normalized=eye_coords_normalized_list,
            pixel_original_ref_approx=eye_coords_pixel_orig_ref_list,
        ),
    }
