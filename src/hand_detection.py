"""
Hand Detection Module
Works with MediaPipe 0.10+ (new API using mp.tasks).
Falls back to legacy mp.solutions if available.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import mediapipe as mp


@dataclass
class HandLandmarks:
    """Stores normalized landmarks for one detected hand."""
    landmarks: List[List[float]]   # 21 x [x, y, z]
    handedness: str                # 'Left' or 'Right'


class HandDetector:
    """
    Detects hands and extracts 21 landmarks per hand.
    Compatible with both old and new MediaPipe versions.
    """

    # ── Landmark indices ──────────────────────────────────────────────────────
    WRIST           = 0
    THUMB_CMC       = 1
    THUMB_MCP       = 2
    THUMB_IP        = 3
    THUMB_TIP       = 4
    INDEX_MCP       = 5
    INDEX_PIP       = 6
    INDEX_DIP       = 7
    INDEX_TIP       = 8
    MIDDLE_MCP      = 9
    MIDDLE_PIP      = 10
    MIDDLE_DIP      = 11
    MIDDLE_TIP      = 12
    RING_MCP        = 13
    RING_PIP        = 14
    RING_DIP        = 15
    RING_TIP        = 16
    PINKY_MCP       = 17
    PINKY_PIP       = 18
    PINKY_DIP       = 19
    PINKY_TIP       = 20

    # Finger tip and pip indices
    FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    FINGER_PIPS = [THUMB_IP,  INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]

    def __init__(self, max_hands: int = 2, detection_confidence: float = 0.7,
                 tracking_confidence: float = 0.5):
        self.max_hands            = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence  = tracking_confidence

        # ── Try new MediaPipe API first, fall back to legacy ──────────────────
        self._use_new_api = False
        try:
            # New API (mediapipe >= 0.10)
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            self._use_new_api = True
            self._setup_new_api()
        except Exception:
            # Legacy API
            self._setup_legacy_api()

    def _setup_new_api(self):
        """Setup using mediapipe.tasks (new API)."""
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._BaseOptions      = mp_python.BaseOptions
        self._HandLandmarker   = mp_vision.HandLandmarker
        self._HandLandmarkerOptions = mp_vision.HandLandmarkerOptions
        self._VisionRunningMode = mp_vision.RunningMode
        self._mp_image         = mp.Image
        self._ImageFormat      = mp.ImageFormat

        # Download model if needed
        import urllib.request, os
        model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        if not os.path.exists(model_path):
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            try:
                urllib.request.urlretrieve(url, model_path)
            except Exception:
                # If download fails, fall back to legacy
                self._use_new_api = False
                self._setup_legacy_api()
                return

        options = self._HandLandmarkerOptions(
            base_options=self._BaseOptions(model_asset_path=model_path),
            running_mode=self._VisionRunningMode.IMAGE,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
        )
        self._landmarker = self._HandLandmarker.create_from_options(options)

    def _setup_legacy_api(self):
        """Setup using mp.solutions.hands (old API)."""
        import mediapipe as mp
        self._mp_hands    = mp.solutions.hands
        self._mp_draw     = mp.solutions.drawing_utils
        self._hands_model = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame_bgr: np.ndarray) -> List[HandLandmarks]:
        """Detect hands in a BGR frame. Returns list of HandLandmarks."""
        if self._use_new_api:
            return self._detect_new(frame_bgr)
        else:
            return self._detect_legacy(frame_bgr)

    def _detect_new(self, frame_bgr: np.ndarray) -> List[HandLandmarks]:
        """Detection using new mediapipe.tasks API."""
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = self._mp_image(
                image_format=self._ImageFormat.SRGB,
                data=rgb
            )
            result = self._landmarker.detect(mp_img)
            hands = []
            if result.hand_landmarks:
                for i, hand_lms in enumerate(result.hand_landmarks):
                    lm_list = [[lm.x, lm.y, lm.z] for lm in hand_lms]
                    handedness = "Right"
                    if result.handedness and i < len(result.handedness):
                        handedness = result.handedness[i][0].display_name
                    hands.append(HandLandmarks(landmarks=lm_list, handedness=handedness))
            return hands
        except Exception:
            return []

    def _detect_legacy(self, frame_bgr: np.ndarray) -> List[HandLandmarks]:
        """Detection using legacy mp.solutions.hands API."""
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self._hands_model.process(rgb)
            hands = []
            if results.multi_hand_landmarks:
                for i, hand_lms in enumerate(results.multi_hand_landmarks):
                    lm_list = [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark]
                    handedness = "Right"
                    if results.multi_handedness and i < len(results.multi_handedness):
                        handedness = results.multi_handedness[i].classification[0].label
                    hands.append(HandLandmarks(landmarks=lm_list, handedness=handedness))
            return hands
        except Exception:
            return []

    def is_finger_up(self, hand: HandLandmarks, finger: str) -> bool:
        """Returns True if the given finger is extended upward."""
        lm = hand.landmarks
        if finger == 'thumb':
            # Thumb: compare tip x vs ip x (horizontal check)
            return lm[self.THUMB_TIP][0] > lm[self.THUMB_IP][0] \
                if hand.handedness == 'Right' \
                else lm[self.THUMB_TIP][0] < lm[self.THUMB_IP][0]
        tip_map = {'index': (self.INDEX_TIP, self.INDEX_PIP),
                   'middle': (self.MIDDLE_TIP, self.MIDDLE_PIP),
                   'ring':   (self.RING_TIP,   self.RING_PIP),
                   'pinky':  (self.PINKY_TIP,  self.PINKY_PIP)}
        if finger not in tip_map:
            return False
        tip_idx, pip_idx = tip_map[finger]
        # Finger up = tip y is ABOVE (smaller) pip y
        return lm[tip_idx][1] < lm[pip_idx][1]

    def count_fingers(self, hand: HandLandmarks) -> int:
        """Count how many fingers are up (0–5)."""
        count = 0
        for f in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            if self.is_finger_up(hand, f):
                count += 1
        return count

    def get_bounding_box(self, hand: HandLandmarks,
                         frame_shape: Tuple[int, int] = (480, 640)) -> Tuple[int, int, int, int]:
        """Returns (x1, y1, x2, y2) bounding box in pixel coords."""
        h, w = frame_shape[:2]
        xs = [lm[0] * w for lm in hand.landmarks]
        ys = [lm[1] * h for lm in hand.landmarks]
        pad = 20
        x1 = max(0, int(min(xs)) - pad)
        y1 = max(0, int(min(ys)) - pad)
        x2 = min(w, int(max(xs)) + pad)
        y2 = min(h, int(max(ys)) + pad)
        return x1, y1, x2, y2