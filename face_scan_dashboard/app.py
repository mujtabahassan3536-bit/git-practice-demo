
"""Local webcam face-detection dashboard built with Flask and OpenCV.

Marriage and Children values are random demo values only.
They are not inferred from a face or camera image.
"""

from __future__ import annotations

import atexit
import random
import threading
import time
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

app = Flask(__name__)


class FaceScanner:
    def __init__(self) -> None:
        self._control_lock = threading.RLock()
        self._frame_lock = threading.Lock()

        self.cap: cv2.VideoCapture | None = None
        self.worker: threading.Thread | None = None

        self.running = False
        self.camera_error = ""
        self.faces: list[tuple[int, int, int, int]] = []

        self.latest_jpeg = self._placeholder_jpeg()
        self.scan_count = 0
        self.last_scan_at = "—"

        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    @staticmethod
    def _placeholder_jpeg() -> bytes:
        frame = np.zeros((480, 720, 3), dtype=np.uint8)

        for y in range(frame.shape[0]):
            shade = int(8 + 12 * y / frame.shape[0])
            frame[y, :] = (shade, shade + 3, shade + 7)

        cv2.rectangle(
            frame,
            (22, 22),
            (698, 458),
            (46, 64, 79),
            2
        )

        cv2.putText(
            frame,
            "CAMERA OFF",
            (218, 220),
            cv2.FONT_HERSHEY_DUPLEX,
            1.25,
            (94, 234, 212),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "Click Start Camera in the dashboard",
            (132, 264),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (190, 205, 218),
            1,
            cv2.LINE_AA
        )

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 82]
        )

        return encoded.tobytes() if ok else b""

    def start(self, camera_index: int = 0) -> tuple[bool, str]:

        with self._control_lock:

            if self.running:
                return True, "Camera is already running."

            if self.face_cascade.empty():
                self.camera_error = (
                    "OpenCV Haar cascade could not be loaded. "
                    "Reinstall the supported opencv-python package."
                )

                return False, self.camera_error

            cap: cv2.VideoCapture | None = None

            try:
                import platform

                if (
                    platform.system() == "Windows"
                    and hasattr(cv2, "CAP_DSHOW")
                ):
                    cap = cv2.VideoCapture(
                        camera_index,
                        cv2.CAP_DSHOW
                    )
                else:
                    cap = cv2.VideoCapture(camera_index)

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

                if not cap.isOpened():

                    cap.release()

                    self.camera_error = (
                        f"Could not open camera {camera_index}. "
                        "Close other apps using the webcam, "
                        "or try another camera index."
                    )

                    return False, self.camera_error

                self.cap = cap
                self.camera_error = ""
                self.running = True

                self.worker = threading.Thread(
                    target=self._capture_loop,
                    name="face-camera",
                    daemon=True
                )

                self.worker.start()

                return True, "Camera started successfully."

            except Exception as exc:

                self.camera_error = f"Camera start failed: {exc}"

                if cap is not None:
                    cap.release()

                self.cap = None
                self.running = False

                return False, self.camera_error

    def _capture_loop(self) -> None:

        while self.running:

            cap = self.cap

            if cap is None:
                break

            ok, frame = cap.read()

            if not ok or frame is None:

                self.camera_error = (
                    "Camera is open but no frame was received. "
                    "Check webcam permissions."
                )

                time.sleep(0.12)
                continue

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            detected = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)
            )

            found = [
                (int(x), int(y), int(w), int(h))
                for x, y, w, h in detected
            ]

            # Draw rectangle around detected faces.
            for x, y, w, h in found:

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (94, 234, 212),
                    2
                )

                cv2.putText(
                    frame,
                    "FACE DETECTED",
                    (x, max(24, y - 11)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (94, 234, 212),
                    2,
                    cv2.LINE_AA
                )

            # Camera header.
            cv2.rectangle(
                frame,
                (0, 0),
                (frame.shape[1], 42),
                (8, 17, 32),
                -1
            )

            cv2.putText(
                frame,
                "LIVE CAMERA | FACE SCANNER",
                (18, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (225, 242, 248),
                1,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                f"Faces: {len(found)}",
                (max(12, frame.shape[1] - 145), 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (94, 234, 212),
                2,
                cv2.LINE_AA
            )

            if not found:

                cv2.putText(
                    frame,
                    "Position your face in the frame",
                    (20, frame.shape[0] - 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (230, 235, 240),
                    1,
                    cv2.LINE_AA
                )

            encoded_ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 84]
            )

            if encoded_ok:

                with self._frame_lock:

                    self.latest_jpeg = encoded.tobytes()
                    self.faces = found
                    self.camera_error = ""

            time.sleep(0.015)

    def stop(self) -> tuple[bool, str]:

        with self._control_lock:

            self.running = False

            worker = self.worker

            if (
                worker
                and worker.is_alive()
                and worker is not threading.current_thread()
            ):
                worker.join(timeout=2.0)

            cap = self.cap

            self.cap = None
            self.worker = None

            if cap is not None:
                cap.release()

            with self._frame_lock:

                self.faces = []
                self.latest_jpeg = self._placeholder_jpeg()

            self.camera_error = ""

            return True, "Camera stopped."

    def frame(self) -> bytes:

        with self._frame_lock:
            return self.latest_jpeg

    def status(self) -> dict[str, Any]:

        with self._frame_lock:
            face_count = len(self.faces)

        return {
            "camera_running": self.running,
            "face_count": face_count,
            "scan_count": self.scan_count,
            "last_scan_at": self.last_scan_at,
            "error": self.camera_error
        }

    def has_face(self) -> bool:

        with self._frame_lock:
            return bool(self.faces)

    def record_scan(self) -> tuple[int, str]:

        self.scan_count += 1
        self.last_scan_at = time.strftime("%H:%M:%S")

        return self.scan_count, self.last_scan_at


# Initialize scanner.
scanner = FaceScanner()

atexit.register(scanner.stop)


# Dashboard route.
@app.route("/")
def dashboard():

    return render_template("index.html")


# Live camera video stream.
@app.route("/video_feed")
def video_feed():

    def generate():

        while True:

            frame = scanner.frame()

            if frame:

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache\r\n\r\n"
                    + frame
                    + b"\r\n"
                )

            time.sleep(0.055)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"}
    )


# Start camera API.
@app.route("/api/camera/start", methods=["POST"])
def camera_start():

    payload = request.get_json(silent=True) or {}

    try:

        camera_index = int(
            payload.get("camera_index", 0)
        )

        if camera_index < 0 or camera_index > 5:
            raise ValueError

    except (TypeError, ValueError):

        return jsonify(
            ok=False,
            message="Invalid camera index."
        ), 400

    ok, message = scanner.start(camera_index)

    return (
        jsonify(
            ok=ok,
            message=message,
            **scanner.status()
        ),
        200 if ok else 500
    )


# Stop camera API.
@app.route("/api/camera/stop", methods=["POST"])
def camera_stop():

    ok, message = scanner.stop()

    return jsonify(
        ok=ok,
        message=message,
        **scanner.status()
    )


# Camera status API.
@app.route("/api/status")
def camera_status():

    return jsonify(scanner.status())


# Marriage and Children demo scan API.
@app.route("/api/predict", methods=["POST"])
def predict_demo():

    if not scanner.running:

        return jsonify(
            ok=False,
            message="Start the camera first."
        ), 400

    if not scanner.has_face():

        return jsonify(
            ok=False,
            message=(
                "No face detected. Position your face "
                "in the camera and try again."
            )
        ), 400

    # Random demo values only.
    # These are NOT actual predictions.
    marriages = random.randint(1, 4)
    children = random.randint(1, 20)

    scan_count, last_scan_at = scanner.record_scan()

    return jsonify(
        ok=True,
        result={
            "marriages": marriages,
            "children": children
        },
        scan_count=scan_count,
        last_scan_at=last_scan_at
    )


# Run Flask application.
if __name__ == "__main__":

    print(
        "Face Scan Dashboard: "
        "http://127.0.0.1:5000"
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )
