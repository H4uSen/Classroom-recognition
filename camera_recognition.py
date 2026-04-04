from app.cli import parse_args, resolve_class_names
from app.runtime import run_camera_preview


if __name__ == "__main__":
    cli_args = parse_args()
    run_camera_preview(
        capture_interval_seconds=cli_args.interval,
        model_path=cli_args.model,
        mqtt_host=cli_args.mqtt_host,
        mqtt_port=cli_args.mqtt_port,
        headless=cli_args.headless,
        class_names=resolve_class_names(cli_args),
    )
