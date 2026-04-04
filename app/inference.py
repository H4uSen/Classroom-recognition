from pathlib import Path

import numpy as np
import tensorflow as tf

from .config import DEFAULT_CLASS_NAMES, MODEL_IMAGE_SIZE, MODEL_PATH, PREDICTION_THRESHOLD, PathLike, PredictionModel


def load_prediction_model(model_path: PathLike = MODEL_PATH) -> PredictionModel:
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
    resolved_class_names = class_names or DEFAULT_CLASS_NAMES
    if len(resolved_class_names) < 2:
        resolved_class_names = DEFAULT_CLASS_NAMES

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
