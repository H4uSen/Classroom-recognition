from collections import deque
import logging

from flask import Flask, Response, jsonify, render_template_string
import cv2
import threading

app = Flask(__name__)

latest_frame = None
latest_prediction = "Sin datos"
recent_mqtt_messages = deque(maxlen=10)

lock = threading.Lock()

def update_data(frame, prediction):
    global latest_frame, latest_prediction
    with lock:
        latest_frame = frame.copy()
        latest_prediction = prediction


def add_mqtt_message(message: str) -> None:
    with lock:
        recent_mqtt_messages.appendleft(message)

def generate_frames():
    global latest_frame
    while True:
        with lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template_string("""
    <html>
    <head>
        <title>Detección</title>
        <style>
            body { text-align: center; font-family: Arial; background: #f5f7fb; color: #1f2937; }
            img { width: 80%; border: 2px solid black; max-width: 920px; }
            h2 { color: #2563eb; }
            .mqtt-box {
                width: 80%;
                max-width: 920px;
                margin: 16px auto;
                padding: 12px;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                background: #ffffff;
                text-align: left;
            }
            .mqtt-box h3 { margin: 0 0 8px 0; color: #0f172a; }
            .mqtt-list {
                margin: 0;
                padding-left: 20px;
                max-height: 220px;
                overflow-y: auto;
                font-family: monospace;
                font-size: 13px;
            }
        </style>
    </head>
    <body>
        <h1>Detección en Tiempo Real</h1>
        <img src="/video">
        <h2 id="pred">Cargando...</h2>
        <div class="mqtt-box">
            <h3>Mensajes MQTT recientes</h3>
            <ol id="mqtt-list" class="mqtt-list"></ol>
        </div>

        <script>
            setInterval(() => {
                fetch('/pred')
                .then(res => res.text())
                .then(data => {
                    document.getElementById('pred').innerText = data;
                });
            }, 500);

            setInterval(() => {
                fetch('/mqtt')
                .then(res => res.json())
                .then(items => {
                    const list = document.getElementById('mqtt-list');
                    list.innerHTML = '';

                    if (!items.length) {
                        const li = document.createElement('li');
                        li.innerText = 'Sin mensajes MQTT todavia';
                        list.appendChild(li);
                        return;
                    }

                    items.forEach(msg => {
                        const li = document.createElement('li');
                        li.innerText = msg;
                        list.appendChild(li);
                    });
                });
            }, 700);
        </script>
    </body>
    </html>
    """)


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/pred')
def pred():
    global latest_prediction
    return latest_prediction


@app.route('/mqtt')
def mqtt_messages():
    with lock:
        return jsonify(list(recent_mqtt_messages))


def start_server():
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app.logger.setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=5000, threaded=True)
