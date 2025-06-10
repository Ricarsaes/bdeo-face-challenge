import torch

DATA_CSV_PATH = "data/data.csv"
IMAGE_DIR = "data/images/"

IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3

BATCH_SIZE = 32
EPOCHS = 25

LEARNING_RATE = 1e-4


MAX_AGE = 100.0

ORIG_IMG_WIDTH = 178.0
ORIG_IMG_HEIGHT = 218.0

VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
