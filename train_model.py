"""
Model Training Script
Collects gesture data and trains a classification model.

Usage:
    python train_model.py --collect   # Collect training images
    python train_model.py --train     # Train the model
    python train_model.py --evaluate  # Evaluate model accuracy
"""

import os
import cv2
import numpy as np
import pickle
import argparse
import time
from pathlib import Path

try:
    import mediapipe as mp
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: sklearn not installed. Run: pip install scikit-learn")

# ─── Config ────────────────────────────────────────────────────────────────

DATASET_DIR  = Path("dataset")
MODEL_PATH   = Path("models/gesture_model.pkl")
LABELS_PATH  = Path("models/label_encoder.pkl")

GESTURES = {
    "hello":    {"label": "Hello",    "samples": 200},
    "yes":      {"label": "Yes",      "samples": 200},
    "no":       {"label": "No",       "samples": 200},
    "thanks":   {"label": "Thank You","samples": 200},
    "peace":    {"label": "Peace",    "samples": 200},
    "please":   {"label": "Please",   "samples": 200},
    "sorry":    {"label": "Sorry",    "samples": 200},
    "namaste":  {"label": "Namaste",  "samples": 150},
}

# ─── Feature Extraction ────────────────────────────────────────────────────

mp_hands = mp.solutions.hands if 'mp' in dir() else None


def extract_features(image_path: str) -> np.ndarray:
    """Extract 63 normalized landmark features (21 points × 3 coords)."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with mp.solutions.hands.Hands(
            static_image_mode=True, max_num_hands=1,
            min_detection_confidence=0.5
        ) as hands:
            results = hands.process(rgb)
            if not results.multi_hand_landmarks:
                return None
            lm = results.multi_hand_landmarks[0]
            coords = []
            xs = [l.x for l in lm.landmark]
            ys = [l.y for l in lm.landmark]
            x_min, y_min = min(xs), min(ys)

            # Normalize relative to wrist
            for l in lm.landmark:
                coords.extend([l.x - x_min, l.y - y_min, l.z])
            return np.array(coords, dtype=np.float32)
    except Exception as e:
        print(f"  Feature extraction failed for {image_path}: {e}")
        return None


# ─── Data Collection ───────────────────────────────────────────────────────

def collect_data():
    """Interactive data collection using webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    gestures = list(GESTURES.keys())
    print("\n=== Data Collection Mode ===")
    print("Press SPACE to start collecting for each gesture.")
    print("Press Q to quit.\n")

    with mp.solutions.hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7
    ) as hands_model:
        for gesture_key in gestures:
            info = GESTURES[gesture_key]
            save_dir = DATASET_DIR / gesture_key
            save_dir.mkdir(parents=True, exist_ok=True)

            existing = len(list(save_dir.glob("*.jpg")))
            needed = info["samples"] - existing
            if needed <= 0:
                print(f"  ✓ {info['label']}: already has {existing} samples, skipping.")
                continue

            print(f"\n[ {info['label'].upper()} ] → Show gesture: {gesture_key}")
            print(f"  Need {needed} more samples (have {existing}).")
            print("  Press SPACE when ready...")

            # Wait for user to press space
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"Gesture: {info['label']}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press SPACE to start", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)
                cv2.imshow("Data Collection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    break
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # Collect samples
            count = 0
            print(f"  Collecting {needed} samples...")
            while count < needed:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)

                # Save every 3rd frame to ensure variety
                if count % 1 == 0:
                    img_path = save_dir / f"{existing + count:04d}.jpg"
                    cv2.imwrite(str(img_path), frame)
                    count += 1

                # Overlay
                progress = f"{count}/{needed}"
                cv2.putText(frame, f"Collecting: {info['label']}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 100), 2)
                cv2.putText(frame, progress, (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 1)
                cv2.imshow("Data Collection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            print(f"  ✓ Saved {count} samples for '{info['label']}'")

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Data collection complete!")


# ─── Training ──────────────────────────────────────────────────────────────

def train_model():
    """Train classifier on collected dataset."""
    if not SKLEARN_AVAILABLE:
        print("ERROR: scikit-learn required. Run: pip install scikit-learn")
        return

    print("\n=== Training Gesture Model ===")
    X, y = [], []

    for gesture_key, info in GESTURES.items():
        gesture_dir = DATASET_DIR / gesture_key
        if not gesture_dir.exists():
            print(f"  WARNING: No data found for '{gesture_key}'. Skipping.")
            continue

        images = list(gesture_dir.glob("*.jpg")) + list(gesture_dir.glob("*.png"))
        print(f"  Processing {len(images)} images for '{info['label']}'...")

        for img_path in images:
            features = extract_features(str(img_path))
            if features is not None:
                X.append(features)
                y.append(info["label"])

    if len(X) < 10:
        print("ERROR: Not enough training data. Run with --collect first.")
        return

    X = np.array(X)
    y = np.array(y)

    print(f"\n  Total samples: {len(X)}")
    print(f"  Classes: {np.unique(y)}")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Train
    print("\n  Training Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Test Accuracy: {acc:.2%}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)
    with open(LABELS_PATH, 'wb') as f:
        pickle.dump(le, f)

    print(f"\n✅ Model saved to {MODEL_PATH}")
    print(f"✅ Labels saved to {LABELS_PATH}")


# ─── ML-based Recognizer ──────────────────────────────────────────────────

class MLGestureRecognizer:
    """
    Optional ML-based gesture recognizer (supplements rule-based system).
    Load the trained model for higher accuracy on custom gestures.
    """

    def __init__(self):
        self.clf = None
        self.le = None
        self.loaded = False
        self._load()

    def _load(self):
        if MODEL_PATH.exists() and LABELS_PATH.exists():
            try:
                with open(MODEL_PATH, 'rb') as f:
                    self.clf = pickle.load(f)
                with open(LABELS_PATH, 'rb') as f:
                    self.le = pickle.load(f)
                self.loaded = True
                print("[MLRecognizer] Model loaded successfully.")
            except Exception as e:
                print(f"[MLRecognizer] Failed to load: {e}")

    def predict(self, landmarks) -> tuple:
        """Returns (label, confidence) or (None, 0)."""
        if not self.loaded or landmarks is None:
            return None, 0.0
        try:
            features = np.array(landmarks, dtype=np.float32).reshape(1, -1)
            proba = self.clf.predict_proba(features)[0]
            idx = np.argmax(proba)
            confidence = float(proba[idx])
            label = self.le.inverse_transform([idx])[0]
            if confidence >= 0.6:
                return label, confidence
        except Exception:
            pass
        return None, 0.0


# ─── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sign Language Translator - Model Training")
    parser.add_argument("--collect",  action="store_true", help="Collect training data")
    parser.add_argument("--train",    action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model accuracy")
    args = parser.parse_args()

    if args.collect:
        collect_data()
    elif args.train:
        train_model()
    elif args.evaluate:
        train_model()  # Re-runs with full eval report
    else:
        parser.print_help()
