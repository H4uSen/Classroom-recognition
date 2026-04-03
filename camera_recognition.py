# Source - https://stackoverflow.com/a/606154
# Posted by John Montgomery, modified by community. See post 'Timeline' for change history
# Retrieved 2026-03-10, License - CC BY-SA 4.0

import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np
import tensorflow as tf
from picamera2 import Picamera2

from mqtt_config import (
    MQTT_DEFAULT_HOST,
    MQTT_DEFAULT_PORT,
    MQTT_TOPIC,
    create_mqtt_client,
    publish_prediction,
)

CAPTURE_INTERVAL_SECONDS = 2.0
MAX_RECENT_CAPTURES = 10
CAPTURE_DIRECTORY = Path(__file__).resolve().parent / "captures"
TRAIN_DATA_DIRECTORY = Path(__file__).resolve().parent / "content" / "train"
MODEL_PATH = Path(__file__).resolve().parent / "modelos_salones.keras"
MODEL_IMAGE_SIZE = (224, 224)
PREDICTION_THRESHOLD = 0.745191216468811
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")
PathLike = Union[str, Path]
PredictionModel = dict[str, Any]
DEFAULT_CAMERA_CANDIDATES = tuple(range(0, 10))
PICAMERA_PREVIEW_SIZE = (640, 480)


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

    suffix = resolved_model_path.suffix.lower()
    if suffix == ".tflite":
        interpreter = tf.lite.Interpreter(model_path=str(resolved_model_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        return {
            "type": "tflite",
            "model": interpreter,
            "input_details": input_details,
            "output_details": output_details,
            "path": resolved_model_path,
        }

    keras_model = tf.keras.models.load_model(resolved_model_path)
    return {
        "type": "keras",
        "model": keras_model,
        "path": resolved_model_path,
    }


def predict_image(
    image_path: PathLike,
    model: PredictionModel,
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

    if model.get("type") == "tflite":
        interpreter = model["model"]
        input_details = model["input_details"]
        output_details = model["output_details"]

        input_dtype = input_details["dtype"]
        input_scale, input_zero_point = input_details.get("quantization", (0.0, 0))

        if input_dtype in (np.int8, np.uint8) and input_scale and input_scale > 0:
            model_input = np.round(img_array / input_scale + input_zero_point).astype(input_dtype)
        else:
            model_input = img_array.astype(input_dtype)

        interpreter.set_tensor(input_details["index"], model_input)
        interpreter.invoke()

        raw_output = interpreter.get_tensor(output_details["index"])
        output_value = float(np.squeeze(raw_output))

        output_dtype = output_details["dtype"]
        output_scale, output_zero_point = output_details.get("quantization", (0.0, 0))
        if output_dtype in (np.int8, np.uint8) and output_scale and output_scale > 0:
            prediction = (output_value - output_zero_point) * output_scale
        else:
            prediction = output_value
    else:
        keras_model = model["model"]
        prediction = float(keras_model.predict(img_array, verbose=0)[0][0])

    prediction = max(0.0, min(1.0, prediction))
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
    camera_index: int | None = None,
    capture_dir: PathLike = CAPTURE_DIRECTORY,
    capture_interval_seconds: float = CAPTURE_INTERVAL_SECONDS,
    max_recent_captures: int = MAX_RECENT_CAPTURES,
    model_path: PathLike = MODEL_PATH,
    threshold: float = PREDICTION_THRESHOLD,
    mqtt_host: str = MQTT_DEFAULT_HOST,
    mqtt_port: int = MQTT_DEFAULT_PORT,
) -> None:
    class_names = get_class_names()
    mqtt_client = None

    try:
        model = load_prediction_model(model_path)
        print(f"Loaded model: {Path(model_path)} ({model['type']})")
        print(f"Class labels: {class_names}")
    except (FileNotFoundError, OSError, ValueError) as exc:
        model = None
        print(f"Model could not be loaded. Captures will continue without prediction. {exc}")

    try:
        mqtt_client = create_mqtt_client(host=mqtt_host, port=mqtt_port)
        print(f"MQTT connected: {mqtt_host}:{mqtt_port} -> topic '{MQTT_TOPIC}'")
    except Exception as exc:
        mqtt_client = None
        print(
            "MQTT not available, publishing disabled. "
            f"Broker={mqtt_host}:{mqtt_port}. {exc}"
        )

    cv2.namedWindow("preview")
    selected_camera_index = 0 if camera_index is None else camera_index

    try:
        picam2 = Picamera2(camera_num=selected_camera_index)
    except Exception as exc:
        print(f"Could not initialize Pi camera (index {selected_camera_index}). {exc}")
        cv2.destroyWindow("preview")
        return

    preview_config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": PICAMERA_PREVIEW_SIZE}
    )
    picam2.configure(preview_config)
    picam2.start()
    time.sleep(0.5)

    try:
        frame_rgb = picam2.capture_array()
        if frame_rgb is None:
            print("Could not capture frames from Pi camera.")
            return

        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        print(f"Using Pi camera index: {selected_camera_index}")

        last_capture_at = time.monotonic()

        while True:
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
                        if mqtt_client is not None:
                            publish_prediction(
                                client=mqtt_client,
                                label=predicted_label,
                                confidence=confidence,
                                raw_score=raw_prediction,
                            )
                except (OSError, tf.errors.OpError, ValueError) as exc:
                    print(exc)
                last_capture_at = now

            frame_rgb = picam2.capture_array()
            if frame_rgb is None:
                continue

            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                break
    finally:
        if mqtt_client is not None:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        picam2.stop()
        picam2.close()
        cv2.destroyWindow("preview")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera preview and classroom occupancy prediction")
    parser.add_argument(
        "--model",
        default=str(MODEL_PATH),
        help="Path to model file (.keras or .tflite)",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Pi camera index to use (default: 0)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=CAPTURE_INTERVAL_SECONDS,
        help="Seconds between captures",
    )
    parser.add_argument(
        "--mqtt-host",
        default=MQTT_DEFAULT_HOST,
        help="MQTT broker host",
    )
    parser.add_argument(
        "--mqtt-port",
        type=int,
        default=MQTT_DEFAULT_PORT,
        help="MQTT broker port",
    )

    cli_args = parser.parse_args()
    run_camera_preview(
        camera_index=cli_args.camera_index,
        capture_interval_seconds=cli_args.interval,
        model_path=cli_args.model,
        mqtt_host=cli_args.mqtt_host,
        mqtt_port=cli_args.mqtt_port,
    )
