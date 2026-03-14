# Source - https://stackoverflow.com/a/606154
# Posted by John Montgomery, modified by community. See post 'Timeline' for change history
# Retrieved 2026-03-10, License - CC BY-SA 4.0

import time
from datetime import datetime
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import tensorflow as tf

CAPTURE_INTERVAL_SECONDS = 2.0
MAX_RECENT_CAPTURES = 10
CAPTURE_DIRECTORY = Path(__file__).resolve().parent / "captures"
TRAIN_DATA_DIRECTORY = Path(__file__).resolve().parent / "content" / "train"
MODEL_PATH = Path(__file__).resolve().parent / "modelos_salones.keras"
MODEL_IMAGE_SIZE = (224, 224)
PREDICTION_THRESHOLD = 0.745191216468811
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")
PathLike = Union[str, Path]


def get_class_names(train_data_dir: PathLike = TRAIN_DATA_DIRECTORY) -> list[str]:
    train_path = Path(train_data_dir)
    if not train_path.exists():
        return ["empty", "full"]

    class_names = sorted(path.name for path in train_path.iterdir() if path.is_dir() and not path.name.startswith("."))
    return class_names or ["empty", "full"]


def load_prediction_model(model_path: PathLike = MODEL_PATH):
    resolved_model_path = Path(model_path)
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Model file not found: {resolved_model_path}")
    return tf.keras.models.load_model(resolved_model_path)


def predict_image(
    image_path: PathLike,
    model,
    class_names: list[str] | None = None,
    image_size: tuple[int, int] = MODEL_IMAGE_SIZE,
    threshold: float = PREDICTION_THRESHOLD,
) -> tuple[str, float, float]:
    resolved_class_names = class_names or ["empty", "full"]
    if len(resolved_class_names) < 2:
        resolved_class_names = ["empty", "full"]

    img = tf.keras.utils.load_img(image_path, target_size=image_size, color_mode="rgb")
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = float(model.predict(img_array, verbose=0)[0][0])
    predicted_index = 1 if prediction > threshold else 0
    predicted_label = resolved_class_names[predicted_index]
    confidence = prediction if predicted_index == 1 else 1.0 - prediction
    return predicted_label, confidence, prediction


def ensure_capture_dir(capture_dir: PathLike = CAPTURE_DIRECTORY) -> Path:
    capture_path = Path(capture_dir)
    capture_path.mkdir(parents=True, exist_ok=True)
    return capture_path


def prune_old_captures(capture_dir: PathLike = CAPTURE_DIRECTORY, keep: int = MAX_RECENT_CAPTURES) -> None:
    capture_path = ensure_capture_dir(capture_dir)
    files = []

    for pattern in IMAGE_EXTENSIONS:
        files.extend(capture_path.glob(pattern))

    files = sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)

    for old_file in files[max(keep, 0):]:
        try:
            old_file.unlink()
        except FileNotFoundError:
            pass


def save_capture(frame, capture_dir: PathLike = CAPTURE_DIRECTORY, max_recent_captures: int = MAX_RECENT_CAPTURES) -> Path:
    capture_path = ensure_capture_dir(capture_dir)
    filename = datetime.now().strftime("capture_%Y%m%d_%H%M%S_%f.jpg")
    output_path = capture_path / filename

    if not cv2.imwrite(str(output_path), frame):
        raise OSError(f"Could not save capture to {output_path}")

    prune_old_captures(capture_path, keep=max_recent_captures)
    return output_path


def run_camera_preview(
    camera_index: int = 0,
    capture_dir: PathLike = CAPTURE_DIRECTORY,
    capture_interval_seconds: float = CAPTURE_INTERVAL_SECONDS,
    max_recent_captures: int = MAX_RECENT_CAPTURES,
    model_path: PathLike = MODEL_PATH,
    threshold: float = PREDICTION_THRESHOLD,
) -> None:
    class_names = get_class_names()

    try:
        model = load_prediction_model(model_path)
        print(f"Loaded model: {Path(model_path)}")
        print(f"Class labels: {class_names}")
    except (FileNotFoundError, OSError, ValueError) as exc:
        model = None
        print(f"Model could not be loaded. Captures will continue without prediction. {exc}")

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(camera_index)

    try:
        if vc.isOpened():
            rval, frame = vc.read()
        else:
            rval = False
            frame = None

        if not rval:
            print("Could not access the camera.")
            return

        last_capture_at = time.monotonic()

        while rval:
            cv2.imshow("preview", frame)

            now = time.monotonic()
            if now - last_capture_at >= capture_interval_seconds:
                try:
                    saved_path = save_capture(frame, capture_dir=capture_dir, max_recent_captures=max_recent_captures)
                    print(f"Saved capture: {saved_path}")

                    if model is not None:
                        predicted_label, confidence, raw_prediction = predict_image(
                            saved_path,
                            model=model,
                            class_names=class_names,
                            threshold=threshold,
                        )
                        print(
                            f"Prediction: {predicted_label} | confidence={confidence:.4f} | raw_score={raw_prediction:.4f}"
                        )
                except (OSError, tf.errors.OpError, ValueError) as exc:
                    print(exc)
                last_capture_at = now

            rval, frame = vc.read()
            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                break
    finally:
        vc.release()
        cv2.destroyWindow("preview")


if __name__ == "__main__":
    run_camera_preview()
