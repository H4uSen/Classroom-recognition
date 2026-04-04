import argparse

from mqtt_config import MQTT_DEFAULT_HOST, MQTT_DEFAULT_PORT

from .config import CAPTURE_INTERVAL_SECONDS, DEFAULT_CLASS_NAMES, MODEL_PATH


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Camera preview and classroom occupancy prediction")
    parser.add_argument(
        "--model",
        default=str(MODEL_PATH),
        help="Path to model file (.keras or .tflite)",
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
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI preview",
    )
    parser.add_argument(
        "--class0",
        default=None,
        help=f"Label for class 0 (default: {DEFAULT_CLASS_NAMES[0]})",
    )
    parser.add_argument(
        "--class1",
        default=None,
        help=f"Label for class 1 (default: {DEFAULT_CLASS_NAMES[1]})",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def resolve_class_names(args: argparse.Namespace) -> list[str]:
    return [
        args.class0 or DEFAULT_CLASS_NAMES[0],
        args.class1 or DEFAULT_CLASS_NAMES[1],
    ]
