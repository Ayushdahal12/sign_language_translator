"""
Camera Manager Module
Handles OpenCV webcam capture and frame processing pipeline.
"""

import cv2
import numpy as np
import threading
import time
from typing import Optional, Tuple


class CameraManager:
    """
    Thread-safe webcam capture manager.
    Provides the latest frame on demand without blocking.
    """

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self.camera_index = camera_index
        self.width = width
        self.height = height

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── Lifecycle ──────────────────────────────

    def start(self) -> bool:
        """Start capturing. Returns True if successful."""
        if self._running:
            return True

        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        """Stop capturing and release resources."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        self._frame = None

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Frame Access ───────────────────────────

    def get_frame(self) -> Optional[np.ndarray]:
        """Returns the latest captured frame (BGR), or None."""
        with self._lock:
            if self._frame is not None:
                return self._frame.copy()
        return None

    def get_frame_rgb(self) -> Optional[np.ndarray]:
        """Returns the latest frame in RGB format."""
        frame = self.get_frame()
        if frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    # ── Frame Processing ──────────────────────

    @staticmethod
    def flip_horizontal(frame: np.ndarray) -> np.ndarray:
        """Mirror the frame (selfie view)."""
        return cv2.flip(frame, 1)

    @staticmethod
    def resize(frame: np.ndarray, width: int, height: int) -> np.ndarray:
        return cv2.resize(frame, (width, height))

    @staticmethod
    def draw_overlay(
        frame: np.ndarray,
        detected_word: Optional[str] = None,
        confidence: float = 0.0,
        hold_progress: float = 0.0,
        sentence: str = "",
        mode: str = "display",
    ) -> np.ndarray:
        """
        Draw a HUD overlay on the frame with detected word and mode info.
        Returns modified frame (does not modify in-place).
        """
        out = frame.copy()
        h, w = out.shape[:2]

        # Semi-transparent top bar
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)

        # Mode label (top-left)
        mode_colors = {
            "display": (255, 200, 50),
            "voice":   (50, 220, 255),
            "typing":  (50, 255, 130),
        }
        mode_color = mode_colors.get(mode, (200, 200, 200))
        cv2.putText(out, f"MODE: {mode.upper()}", (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 1, cv2.LINE_AA)

        # Confidence (top-right)
        if confidence > 0:
            conf_text = f"Conf: {confidence:.0%}"
            cv2.putText(out, conf_text, (w - 140, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        # Detected word (center)
        if detected_word:
            text_size = cv2.getTextSize(detected_word, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 2)[0]
            tx = (w - text_size[0]) // 2
            cv2.putText(out, detected_word, (tx, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)

        # Hold progress bar (bottom of frame)
        if hold_progress > 0:
            bar_w = int(w * hold_progress)
            cv2.rectangle(out, (0, h - 6), (w, h), (40, 40, 40), -1)
            color = (50, 255, 130) if hold_progress < 1.0 else (50, 220, 255)
            cv2.rectangle(out, (0, h - 6), (bar_w, h), color, -1)

        # Sentence strip (bottom overlay)
        if sentence and mode == "typing":
            overlay2 = out.copy()
            cv2.rectangle(overlay2, (0, h - 50), (w, h - 6), (0, 0, 0), -1)
            cv2.addWeighted(overlay2, 0.5, out, 0.5, 0, out)
            cv2.putText(out, sentence, (12, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1, cv2.LINE_AA)

        return out

    # ── Private ────────────────────────────────

    def _capture_loop(self):
        while self._running and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                frame = cv2.flip(frame, 1)  # Mirror by default
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.01)

    def __del__(self):
        self.stop()
