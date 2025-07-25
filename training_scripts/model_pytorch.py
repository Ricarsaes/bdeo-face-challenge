# model_pytorch.py
"""
Defines a multi-task PyTorch neural network model for face attribute prediction
using MobileNetV2 as the backbone. The model simultaneously predicts age (regression),
gender (binary classification), and eye coordinates (regression) from face images.
"""
import torch
from torch import nn
from torchvision import models

from config import IMG_HEIGHT, IMG_WIDTH


class FaceMultitaskModel(nn.Module):
    def __init__(self, num_age_outputs=1, num_gender_outputs=1, num_eye_outputs=4):
        super().__init__()
        self.base_model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )

        # Remove the original classifier

        num_ftrs = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Identity()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout_shared = nn.Dropout(0.3)

        # Age head
        self.age_fc1 = nn.Linear(num_ftrs, 128)
        self.age_dropout = nn.Dropout(0.2)
        self.age_output = nn.Linear(128, num_age_outputs)

        # Gender head
        self.gender_fc1 = nn.Linear(num_ftrs, 64)
        self.gender_dropout = nn.Dropout(0.2)
        self.gender_output = nn.Linear(64, num_gender_outputs)

        # Eyes head
        self.eyes_fc1 = nn.Linear(num_ftrs, 256)
        self.eyes_fc2 = nn.Linear(256, 128)
        self.eyes_dropout = nn.Dropout(0.3)
        self.eyes_output = nn.Linear(128, num_eye_outputs)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x_shared = self.dropout_shared(x)

        # Age branch
        age = self.relu(self.age_fc1(x_shared))
        age = self.age_dropout(age)
        age_pred = self.age_output(age)

        # Gender branch
        gender = self.relu(self.gender_fc1(x_shared))
        gender = self.gender_dropout(gender)
        gender_pred = self.gender_output(gender)

        # Eyes branch
        eyes = self.relu(self.eyes_fc1(x_shared))
        eyes = self.eyes_dropout(eyes)
        eyes = self.relu(self.eyes_fc2(eyes))
        eyes_pred = self.eyes_output(eyes)

        return age_pred, gender_pred, eyes_pred


if __name__ == "__main__":
    model = FaceMultitaskModel()
    print(model)
    # Test with a dummy input
    dummy_input = torch.randn(2, 3, IMG_HEIGHT, IMG_WIDTH)
    age_p, gender_p, eyes_p = model(dummy_input)
    print("Age output shape:", age_p.shape)
    print("Gender output shape:", gender_p.shape)
    print("Eyes output shape:", eyes_p.shape)
