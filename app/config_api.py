import torch

MODEL_STATE_DICT_PATH = "models_pytorch/face_multitask_model_pytorch_best.pth"

MODEL_MODULE = "app.models_pytorch.model_pytorch"
MODEL_CLASS_NAME = "FaceMultitaskModel"

IMG_HEIGHT = 224
IMG_WIDTH = 224

MAX_AGE = 100.0

ORIG_IMG_WIDTH = 178.0
ORIG_IMG_HEIGHT = 218.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
