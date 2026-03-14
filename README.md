# 🤟 Sign Language Translator
 Computer Vision & NLP

A real-time sign language translator that detects hand gestures via webcam and converts them into **text and speech output** — built with Python, MediaPipe, and Streamlit.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Real-time detection** | 30 FPS gesture recognition via webcam |
| **11 built-in gestures** | Hello, Yes, No, Thank You, Peace, Sorry, Please, OK, Namaste, Stop, Bad |
| **3 output modes** | Display Only · Text + Voice · Typing Mode |
| **Sentence builder** | Hold gestures to build full sentences |
| **Offline speech** | pyttsx3 — no internet needed |
| **Beautiful UI** | Dark-mode Streamlit interface |
| **ML training pipeline** | Collect data & train custom gesture classifier |

---



### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. Open browser
Navigate to `http://localhost:8501`

---

##  Gesture Reference

| Gesture | How to do it |
|---|---|
| 👋 **Hello** | Raise only your index finger |
| 👍 **Yes** | Thumbs up |
| ✊ **No** | Make a fist |
| ✋ **Stop** | All 5 fingers, palm facing camera |
| 🙏 **Thank You** | Open palm, fingers spread upward |
| ✌️ **Peace** | Index + middle fingers up (V shape) |
| 🤲 **Please** | Four fingers up, thumb down |
| 🙇 **Sorry** | Three fingers up |
| 👌 **OK** | Touch thumb and index fingertip in a circle |
| 🙏 **Namaste** | Both hands joined together |
| 👎 **Bad** | Thumb pointing downward |

---

## 📁 Project Structure

```
sign_language_translator/
│
├── app.py                      ← Main entry point
├── train_model.py              ← Data collection & ML training
├── requirements.txt
│
├── src/
│   ├── hand_detection.py       ← MediaPipe wrapper (21 landmarks)
│   ├── gesture_recognition.py  ← Rule-based gesture classifier
│   ├── sentence_builder.py     ← Debounced sentence construction
│   └── speech_output.py        ← pyttsx3 offline TTS
│
├── ui/
│   └── interface.py            ← Streamlit UI (dark mode)
│
├── models/
│   ├── gesture_model.pkl       ← Trained ML model (after training)
│   └── label_encoder.pkl       ← Label encoder
│
└── dataset/
    ├── hello/                  ← Training images per gesture
    ├── yes/
    ├── no/
    └── ...
```

---

## 🛠️ How It Works

### Detection Pipeline
```
Webcam → OpenCV → MediaPipe (21 landmarks) → Rule Engine → Word → Output
```

### MediaPipe Landmarks
MediaPipe provides 21 landmarks per hand. The system uses:
- **Finger tip positions** (indices 4, 8, 12, 16, 20)
- **Finger PIP joints** (indices 6, 10, 14, 18) for bend detection
- **Wrist** (index 0) as the base reference point
- **Inter-landmark distances** for special gestures (OK, Namaste)

### Rule Engine (gesture_recognition.py)
Each gesture is a Python function that evaluates landmark geometry:
```python
# Example: "Hello" rule
def _rule_hello(self, hands):
    h = hands[0]
    index_up  = detector.is_finger_up(h, 'index')   # True
    middle_up = detector.is_finger_up(h, 'middle')  # False
    ring_up   = detector.is_finger_up(h, 'ring')    # False
    if index_up and not middle_up and not ring_up:
        return ("Hello", 0.90)
```

### Typing Mode (sentence_builder.py)
- Gesture must be held for **1.2 seconds** (configurable) before being added
- A progress bar fills up while you hold the gesture
- Press "Clear Sentence" to reset

---

 Training Your Own Gestures

### Step 1: Collect training data
```bash
python train_model.py --collect
```
Follow the prompts — collect ~200 images per gesture.

### Step 2: Train the model
```bash
python train_model.py --train
```

### Step 3: Evaluate
```bash
python train_model.py --evaluate
```

The trained model is saved to `models/gesture_model.pkl` and automatically loaded by the app.

---

## Technical Specifications

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| Computer Vision | OpenCV 4.8+ |
| Hand Tracking | MediaPipe 0.10+ |
| Classification | Rule-based + Random Forest |
| Speech | pyttsx3 (offline) |
| UI Framework | Streamlit 1.28+ |
| Feature Vector | 63-dim (21 landmarks × 3 coords) |
| Input | 640×480 webcam @ 30 FPS |

