import json
from datetime import datetime, timezone
from typing import Any

try:
	import paho.mqtt.client as mqtt
except ModuleNotFoundError:
	mqtt = None

MQTT_TOPIC = "class/prediction"
MQTT_DEFAULT_HOST = "localhost"
MQTT_DEFAULT_PORT = 1883


def create_mqtt_client(
	host: str = MQTT_DEFAULT_HOST,
	port: int = MQTT_DEFAULT_PORT,
	client_id: str = "classroom-recognition",
):
	if mqtt is None:
		raise RuntimeError("paho-mqtt is not installed in the active Python environment")
	client = mqtt.Client(client_id=client_id)
	try:
		client.connect(host, port, keepalive=60)
	except OSError as exc:
		raise ConnectionError(
			f"Could not connect to MQTT broker at {host}:{port}. "
			"Check that the broker is running and reachable."
		) from exc
	client.loop_start()
	return client


def publish_prediction(
	client: Any,
	label: str,
	confidence: float,
	raw_score: float,
	topic: str = MQTT_TOPIC,
) -> None:
	payload = {
 		"timestamp": datetime.now(timezone.utc).isoformat(),
		"label": label,
		"confidence": float(confidence),
		"raw_score": float(raw_score),
	}
	client.publish(topic, json.dumps(payload), qos=0, retain=False)
