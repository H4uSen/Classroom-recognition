from datetime import datetime
from pathlib import Path

import cv2

from .config import CAPTURE_DIRECTORY, IMAGE_EXTENSIONS, MAX_RECENT_CAPTURES, PathLike


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
