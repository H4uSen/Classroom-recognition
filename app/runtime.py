import threading
import time
from collections import deque
from datetime import datetime

import cv2
import tensorflow as tf

from mqtt_config import MQTT_DEFAULT_HOST, MQTT_DEFAULT_PORT, MQTT_TOPIC, create_mqtt_client, publish_prediction
from web_server import start_server, update_data

from .camera import capture_bgr_frame, open_pi_camera
from .capture_store import save_capture
from .config import CAPTURE_DIRECTORY, CAPTURE_INTERVAL_SECONDS, DEFAULT_CLASS_NAMES, MAX_RECENT_CAPTURES, MODEL_PATH, PREDICTION_THRESHOLD, PathLike
from .inference import load_prediction_model, predict_image


def run_camera_preview(
    capture_dir: PathLike = CAPTURE_DIRECTORY,
    capture_interval_seconds: float = CAPTURE_INTERVAL_SECONDS,
    max_recent_captures: int = MAX_RECENT_CAPTURES,
    model_path: PathLike = MODEL_PATH,
    threshold: float = PREDICTION_THRESHOLD,
    mqtt_host: str = MQTT_DEFAULT_HOST,
    mqtt_port: int = MQTT_DEFAULT_PORT,
    headless: bool = False,
    class_names: list[str] | None = None,
) -> None:
    max_console_predictions = 10
    resolved_class_names = class_names or DEFAULT_CLASS_NAMES
    if len(resolved_class_names) < 2:
        resolved_class_names = DEFAULT_CLASS_NAMES

    mqtt_client = None
    model = None

    try:
        model = load_prediction_model(model_path)
        print(f"Loaded model: {model_path} ({model['type']})")
        print(f"Class labels: {resolved_class_names}")

        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
        print("Web server running on port 5000")
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Model could not be loaded. Captures will continue without prediction. {exc}")

    try:
        mqtt_client = create_mqtt_client(host=mqtt_host, port=mqtt_port)
        print(f"MQTT connected: {mqtt_host}:{mqtt_port} -> topic '{MQTT_TOPIC}'")
    except Exception as exc:
        print(
            "MQTT not available, publishing disabled. "
            f"Broker={mqtt_host}:{mqtt_port}. {exc}"
        )

    if not headless:
        cv2.namedWindow("preview")

    selected_camera_index = 0

    try:
        picam2 = open_pi_camera(selected_camera_index)
    except Exception as exc:
        print(f"Could not initialize Pi camera (index {selected_camera_index}). {exc}")
        if not headless:
            cv2.destroyWindow("preview")
        return

    try:
        frame = capture_bgr_frame(picam2)
        if frame is None:
            print("Could not capture frames from Pi camera.")
            return

        print(f"Using Pi camera index: {selected_camera_index}")

        recent_predictions = deque(maxlen=max_console_predictions)
        printed_panel_lines = 0

        def render_prediction_panel() -> None:
            nonlocal printed_panel_lines
            if printed_panel_lines:
                print(f"\033[{printed_panel_lines}F", end="")

            lines = [f"Recent predictions (max {max_console_predictions}):"]
            entries = list(recent_predictions)
            for idx in range(max_console_predictions):
                if idx < len(entries):
                    lines.append(f"{idx + 1:02d}. {entries[idx]}")
                else:
                    lines.append(f"{idx + 1:02d}. -")

            for line in lines:
                print(f"\033[2K{line}")

            printed_panel_lines = len(lines)

        render_prediction_panel()

        last_capture_at = time.monotonic()
        ultima_prediccion = "Sin datos"

        while True:
            if not headless:
                cv2.imshow("preview", frame)

            now = time.monotonic()
            if now - last_capture_at >= capture_interval_seconds:
                try:
                    saved_path = save_capture(frame, capture_dir=capture_dir, max_recent_captures=max_recent_captures)
                    #print(f"Saved capture: {saved_path}")

                    if model is not None:
                        predicted_label, confidence, raw_prediction = predict_image(
                            saved_path,
                            model=model,
                            class_names=resolved_class_names,
                            threshold=threshold,
                        )
                        ultima_prediccion = f"{predicted_label} ({confidence:.2f})"
                        recent_predictions.append(
                            f"{datetime.now().strftime('%H:%M:%S')} | {predicted_label} | confidence={confidence:.4f} | raw={raw_prediction:.4f}"
                        )
                        render_prediction_panel()
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

            frame = capture_bgr_frame(picam2)
            if frame is None:
                continue

            cv2.putText(frame, ultima_prediccion, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            update_data(frame, ultima_prediccion if model is not None else "Sin prediccion")

            if not headless:
                key = cv2.waitKey(20)
                if key == 27:
                    break
            else:
                time.sleep(0.02)
    finally:
        if mqtt_client is not None:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        picam2.stop()
        picam2.close()
        if not headless:
            cv2.destroyWindow("preview")
