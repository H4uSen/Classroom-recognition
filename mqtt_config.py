import json
from datetime import datetime, timezone
from typing import Any

try:
	import paho.mqtt.client as mqtt
except ModuleNotFoundError:
	mqtt = None

MQTT_TOPIC = "class/prediction"
MQTT_DEFAULT_HOST = "hausen.local"
MQTT_DEFAULT_PORT = 7474


def create_mqtt_client(
	host: str = MQTT_DEFAULT_HOST,
	port: int = MQTT_DEFAULT_PORT,
	client_id: str = "classroom-recognition",
):
	if mqtt is None:
		raise RuntimeError("paho-mqtt no está instalado o no se pudo importar en el ambiente actual.")
	client = mqtt.Client(client_id=client_id)
	try:
		client.connect(host, port, keepalive=60)
	except OSError as exc:
		raise ConnectionError(
			f"No se pudo conectar al broker MQTT en {host}:{port}. "
			"Verifique que el broker esté en ejecución y sea alcanzable."
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
 		"fecha": datetime.now(timezone.utc).isoformat(),
		"prediccion": label,
		"certeza": float(confidence),
		"prediccion_cruda": float(raw_score),
	}
	client.publish(topic, json.dumps(payload), qos=0, retain=True)
