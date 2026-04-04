from flask import Flask, Response, render_template_string
import cv2
import threading

app = Flask(__name__)

latest_frame = None
latest_prediction = "Sin datos"

lock = threading.Lock()

def update_data(frame, prediction):
    global latest_frame, latest_prediction
    with lock:
        latest_frame = frame.copy()
        latest_prediction = prediction

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
            body { text-align: center; font-family: Arial; }
            img { width: 80%; border: 2px solid black; }
            h2 { color: blue; }
        </style>
    </head>
    <body>
        <h1>Detección en Tiempo Real</h1>
        <img src="/video">
        <h2 id="pred">Cargando...</h2>

        <script>
            setInterval(() => {
                fetch('/pred')
                .then(res => res.text())
                .then(data => {
                    document.getElementById('pred').innerText = data;
                });
            }, 500);
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


def start_server():
    app.run(host='0.0.0.0', port=5000, threaded=True)
