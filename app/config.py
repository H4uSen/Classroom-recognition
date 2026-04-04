from pathlib import Path
from typing import Any, Union

CAPTURE_INTERVAL_SECONDS = 1
MAX_RECENT_CAPTURES = 10
CAPTURE_DIRECTORY = Path(__file__).resolve().parent.parent / "captures"
MODEL_PATH = Path(__file__).resolve().parent.parent / "modelos_salones.keras"
MODEL_IMAGE_SIZE = (224, 224)
PREDICTION_THRESHOLD = 0.745191216468811
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")
DEFAULT_CLASS_NAMES = ["empty", "full"]
DEFAULT_CAMERA_CANDIDATES = tuple(range(0, 10))
PICAMERA_PREVIEW_SIZE = (640, 480)

PathLike = Union[str, Path]
PredictionModel = dict[str, Any]
