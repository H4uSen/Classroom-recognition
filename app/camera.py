import time

import cv2
from picamera2 import Picamera2

from .config import PICAMERA_PREVIEW_SIZE


def open_pi_camera(camera_index: int):
    picam2 = Picamera2(camera_num=camera_index)
    preview_config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": PICAMERA_PREVIEW_SIZE}
    )
    picam2.configure(preview_config)
    picam2.start()
    time.sleep(0.5)
    return picam2


def capture_bgr_frame(picam2) -> cv2.typing.MatLike | None:
    frame_rgb = picam2.capture_array()
    if frame_rgb is None:
        return None
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
