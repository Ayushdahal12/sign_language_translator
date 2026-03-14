"""
Sign Language Translator — Streamlit UI
Beautiful, modern interface for the final year project.
"""

import streamlit as st
import cv2
import numpy as np
import time
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.hand_detection import HandDetector
from src.gesture_recognition import GestureRecognizer
from src.sentence_builder import SentenceBuilder
from src.speech_output import SpeechController


def run_app():
    st.set_page_config(
        page_title="Sign Language Translator",
        page_icon="🤟",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()
    _render_sidebar()
    _render_main()


def _inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    .stApp { background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0d1117 100%); }

    .main-title {
        font-size: 2.8rem; font-weight: 700;
        background: linear-gradient(135deg, #6ee7f7 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.2rem; letter-spacing: -0.02em;
    }
    .sub-title {
        text-align: center; color: #64748b; font-size: 0.95rem;
        font-weight: 400; margin-bottom: 2rem;
    }

    /* Mode toggle pill */
    .mode-toggle-wrap {
        display: flex; gap: 0.5rem; margin: 0.5rem 0 1.2rem;
        justify-content: center;
    }
    .mode-pill {
        flex: 1; text-align: center; padding: 0.55rem 0.5rem;
        border-radius: 10px; font-weight: 700; font-size: 0.88rem;
        cursor: pointer; transition: all 0.2s;
        border: 2px solid #2d3748; color: #64748b; background: #1a2035;
    }
    .mode-pill-words.active {
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        border-color: #6366f1; color: white;
    }
    .mode-pill-alpha.active {
        background: linear-gradient(135deg, #10b981, #06b6d4);
        border-color: #06b6d4; color: white;
    }

    .detection-card {
        background: linear-gradient(135deg, #1e2433 0%, #1a2035 100%);
        border: 1px solid #2d3748; border-radius: 16px;
        padding: 1.5rem; text-align: center; margin: 0.5rem 0;
        position: relative; overflow: hidden;
    }
    .detection-card::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #6ee7f7, #a78bfa, #f472b6);
        border-radius: 16px 16px 0 0;
    }
    .detection-label {
        color: #64748b; font-size: 0.75rem; font-weight: 600;
        letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.5rem;
    }
    .detection-word {
        font-size: 3rem; font-weight: 700; color: #e2e8f0;
        line-height: 1; min-height: 3.5rem;
        display: flex; align-items: center; justify-content: center;
    }
    .detection-emoji { font-size: 2.5rem; margin-top: 0.5rem; }

    .conf-bar-bg {
        background: #1a202c; border-radius: 8px; height: 8px;
        margin: 0.75rem 0 0.25rem; overflow: hidden;
    }
    .conf-bar-fill {
        height: 100%; border-radius: 8px;
        background: linear-gradient(90deg, #6ee7f7, #a78bfa);
        transition: width 0.3s ease;
    }

    /* Typing output box */
    .typing-box {
        background: #0d1117; border: 2px solid #2d3748; border-radius: 12px;
        padding: 1rem 1.25rem; font-family: 'JetBrains Mono', monospace;
        font-size: 1.3rem; color: #a3e635; min-height: 4rem;
        word-break: break-all; letter-spacing: 0.05em; line-height: 1.6;
    }
    .typing-box-alpha {
        border-color: #06b6d4;
    }
    .typing-box-words {
        border-color: #6366f1;
    }
    .typing-label {
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
        text-transform: uppercase; margin-bottom: 0.4rem; margin-top: 0.8rem;
    }

    .stat-card {
        background: #1a2035; border: 1px solid #2d3748;
        border-radius: 10px; padding: 0.85rem 1rem; text-align: center;
    }
    .stat-value { font-size: 1.6rem; font-weight: 700; color: #e2e8f0; }
    .stat-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; }

    .gesture-card {
        background: #1a2035; border: 1px solid #2d3748; border-radius: 10px;
        padding: 0.75rem 1rem; margin: 0.35rem 0;
        display: flex; align-items: center; gap: 0.75rem;
    }
    .gesture-emoji-sm { font-size: 1.4rem; }
    .gesture-word { color: #e2e8f0; font-weight: 600; font-size: 0.9rem; }
    .gesture-how { color: #64748b; font-size: 0.75rem; }

    .alpha-grid {
        display: grid; grid-template-columns: 1fr 1fr; gap: 0.25rem; margin-top: 0.25rem;
    }
    .alpha-card {
        background: #1a2035; border: 1px solid #2d3748; border-radius: 8px;
        padding: 0.4rem 0.55rem; display: flex; align-items: center; gap: 0.4rem;
    }
    .alpha-letter { font-size: 1.05rem; font-weight: 800; color: #6ee7f7; min-width: 1rem; }
    .alpha-emoji  { font-size: 1.15rem; }
    .alpha-how    { color: #64748b; font-size: 0.63rem; line-height: 1.3; }

    .guide-section {
        color: #a78bfa; font-size: 0.68rem; font-weight: 700;
        letter-spacing: 0.12em; text-transform: uppercase; margin: 0.65rem 0 0.3rem;
    }

    .info-pill {
        background: #1e3a5f; border: 1px solid #2563eb; border-radius: 8px;
        padding: 0.6rem 1rem; color: #93c5fd; font-size: 0.82rem;
        text-align: center; margin: 0.5rem 0;
    }

    .camera-placeholder {
        background: #111827; border: 2px dashed #2d3748; border-radius: 12px;
        min-height: 380px; display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        color: #4b5563; font-size: 0.9rem; gap: 0.5rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #111827 100%);
        border-right: 1px solid #1e2433;
    }
    [data-testid="stSidebar"] .stButton button {
        width: 100%; border-radius: 8px; font-weight: 600;
        letter-spacing: 0.04em; font-size: 0.85rem; padding: 0.5rem 1rem;
        border: 1px solid #2d3748; background: #1a2035; color: #e2e8f0; transition: all 0.2s;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background: #2d3748; border-color: #4b5563;
    }
    .stRadio label { color: #94a3b8 !important; font-size: 0.88rem !important; }
    .stSlider [data-testid="stMarkdown"] { color: #94a3b8 !important; }
    label { color: #94a3b8 !important; }
    .stCheckbox label { color: #94a3b8 !important; }
    </style>
    """, unsafe_allow_html=True)


def _render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 1rem 0 0.5rem;'>
            <span style='font-size:2.5rem'>🤟</span>
            <div style='font-size:1.1rem; font-weight:700; color:#e2e8f0; margin-top:0.3rem;'>SL Translator</div>
            <div style='font-size:0.72rem; color:#64748b; letter-spacing:0.1em;'>FINAL YEAR PROJECT</div>
        </div>
        <hr style='border-color:#1e2433; margin: 1rem 0;'>
        """, unsafe_allow_html=True)

        # ── Recognition Mode (Words vs Alphabet) ──
        st.markdown("<div style='color:#94a3b8; font-size:0.75rem; font-weight:600; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:0.4rem;'>Recognition Mode</div>", unsafe_allow_html=True)
        rec_mode = st.radio(
            "rec_mode",
            options=["💬 Word Mode", "🔤 Alphabet Mode"],
            index=st.session_state.get("rec_mode_index", 0),
            label_visibility="collapsed",
        )
        st.session_state["rec_mode"] = "words" if "Word" in rec_mode else "alphabet"
        st.session_state["rec_mode_index"] = 0 if "Word" in rec_mode else 1

        if st.session_state["rec_mode"] == "words":
            st.markdown("<div style='background:#1e2a4a; border:1px solid #3b82f6; border-radius:8px; padding:0.5rem 0.75rem; color:#93c5fd; font-size:0.78rem; margin-top:0.3rem;'>✅ Detects: Hello, Yes, No, OK, Peace, Stop...</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background:#0d2e2a; border:1px solid #10b981; border-radius:8px; padding:0.5rem 0.75rem; color:#6ee7b7; font-size:0.78rem; margin-top:0.3rem;'>✅ Detects: A B C D E F G ... Z</div>", unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#1e2433; margin: 1rem 0;'>", unsafe_allow_html=True)

        # ── Output Mode ──
        st.markdown("<div style='color:#94a3b8; font-size:0.75rem; font-weight:600; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:0.5rem;'>Output Mode</div>", unsafe_allow_html=True)
        mode = st.radio(
            "mode",
            options=["Display Only", "Text + Voice", "Typing Mode"],
            index=st.session_state.get("mode_index", 0),
            label_visibility="collapsed",
        )
        st.session_state["mode"] = mode
        st.session_state["mode_index"] = ["Display Only", "Text + Voice", "Typing Mode"].index(mode)

        st.markdown("<hr style='border-color:#1e2433; margin: 1rem 0;'>", unsafe_allow_html=True)

        # ── Camera ──
        st.markdown("<div style='color:#94a3b8; font-size:0.75rem; font-weight:600; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:0.5rem;'>Camera</div>", unsafe_allow_html=True)
        cam_running = st.session_state.get("camera_running", False)
        if not cam_running:
            if st.button("▶  Start Camera", use_container_width=True):
                st.session_state["camera_running"] = True
                st.session_state["session_start"] = time.time()
                st.rerun()
        else:
            if st.button("⏹  Stop Camera", use_container_width=True):
                st.session_state["camera_running"] = False
                st.rerun()

        st.markdown("<hr style='border-color:#1e2433; margin: 1rem 0;'>", unsafe_allow_html=True)

        # ── Settings ──
        st.markdown("<div style='color:#94a3b8; font-size:0.75rem; font-weight:600; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:0.5rem;'>Settings</div>", unsafe_allow_html=True)
        show_landmarks  = st.toggle("Show hand landmarks", value=True)
        show_confidence = st.toggle("Show confidence bar", value=True)
        st.session_state["show_landmarks"]  = show_landmarks
        st.session_state["show_confidence"] = show_confidence
        hold_time = st.slider("Gesture hold time (sec)", 0.5, 3.0, 1.2, 0.1)
        st.session_state["hold_time"] = hold_time

        st.markdown("<hr style='border-color:#1e2433; margin: 1rem 0;'>", unsafe_allow_html=True)

        # ── Gesture Guide ──
        st.markdown("<div style='color:#94a3b8; font-size:0.75rem; font-weight:600; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:0.3rem;'>Gesture Guide</div>", unsafe_allow_html=True)

        st.markdown("<div class='guide-section'>💬 Common Words</div>", unsafe_allow_html=True)
        for emoji, word, how in [
            ("👋","Hello","Index finger up"),
            ("👍","Yes","Thumb up"),
            ("✊","No","Tight fist"),
            ("🖐️","Stop","All 5 fingers open"),
            ("🤚","Thank You","Open palm up"),
            ("✌️","Peace","Index+middle up"),
            ("🖖","I Love You","Four fingers up"),
            ("🤟","Please","Thumb+index+pinky up"),
            ("🤟","Sorry","Three fingers up"),
            ("👌","OK","Thumb+index circle"),
            ("🤲","Namaste","Both hands joined"),
            ("👎","Bad","Thumb down"),
        ]:
            st.markdown(f"""
            <div class='gesture-card'>
                <span class='gesture-emoji-sm'>{emoji}</span>
                <div><div class='gesture-word'>{word}</div><div class='gesture-how'>{how}</div></div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='guide-section'>🔤 ASL Alphabet A – Z</div>", unsafe_allow_html=True)
        st.markdown("<div class='alpha-grid'>", unsafe_allow_html=True)
        for letter, emoji, how in [
            ("A","✊","Fist, thumb on side"), ("B","🤚","4 fingers up, thumb in"),
            ("C","🤏","Curved C shape"),     ("D","☝️","Index up, thumb touches middle"),
            ("E","✋","All fingers bent down"),("F","👌","Index+thumb circle, rest up"),
            ("G","👉","Index points sideways"),("H","🤞","Index+middle point sideways"),
            ("I","🤙","Pinky only up"),       ("J","🤙","Pinky up + draw J"),
            ("K","✌️","V + thumb between"),  ("L","👆","L shape index+thumb"),
            ("M","✊","3 fingers over thumb"),("N","✊","2 fingers over thumb"),
            ("O","👌","All fingers form O"),  ("P","👇","K shape pointing down"),
            ("Q","👇","G shape pointing down"),("R","🤞","Index+middle crossed"),
            ("S","✊","Thumb over fingers"),  ("T","✊","Thumb between fingers"),
            ("U","✌️","Index+middle together"),("V","✌️","V shape spread"),
            ("W","🖖","3 fingers spread up"), ("X","☝️","Index finger hooked"),
            ("Y","🤙","Thumb+pinky out"),     ("Z","☝️","Index draws Z"),
        ]:
            st.markdown(f"""
            <div class='alpha-card'>
                <span class='alpha-letter'>{letter}</span>
                <span class='alpha-emoji'>{emoji}</span>
                <div class='alpha-how'>{how}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def _render_main():
    st.markdown("""
    <div class='main-title'>Sign Language Translator</div>
    <div class='sub-title'>Real-time hand gesture recognition · Text & Voice output</div>
    """, unsafe_allow_html=True)

    col_cam, col_info = st.columns([3, 2], gap="large")
    with col_cam:
        _render_camera_section()
    with col_info:
        _render_info_section()


def _render_camera_section():
    mode        = st.session_state.get("mode", "Display Only")
    rec_mode    = st.session_state.get("rec_mode", "words")
    cam_running = st.session_state.get("camera_running", False)

    camera_placeholder = st.empty()

    if not cam_running:
        camera_placeholder.markdown("""
        <div class='camera-placeholder'>
            <span style='font-size:3rem;'>📷</span>
            <div style='font-weight:600; color:#64748b;'>Camera not started</div>
            <div style='font-size:0.8rem;'>Click "Start Camera" in the sidebar</div>
        </div>""", unsafe_allow_html=True)
        return

    detector   = HandDetector(max_hands=2)
    recognizer = GestureRecognizer(mode=rec_mode)
    builder    = SentenceBuilder(hold_time=st.session_state.get("hold_time", 1.2))
    speech     = SpeechController(cooldown_seconds=2.5)

    if mode not in ("Text + Voice",):
        speech.enabled = False

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    show_landmarks = st.session_state.get("show_landmarks", True)
    frame_count = 0

    while st.session_state.get("camera_running", False):

        # Update recognizer mode live if user switches
        new_rec_mode = st.session_state.get("rec_mode", "words")
        recognizer.set_mode(new_rec_mode)

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        hands = detector.detect(frame)

        if show_landmarks and hands:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            import mediapipe as mp
            mp_hands  = mp.solutions.hands
            mp_draw   = mp.solutions.drawing_utils
            mp_styles = mp.solutions.drawing_styles
            with mp_hands.Hands(static_image_mode=True, max_num_hands=2) as tmp:
                res = tmp.process(rgb)
                if res.multi_hand_landmarks:
                    for lm in res.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            frame, lm, mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style(),
                        )

        result       = recognizer.recognize(hands)
        current_word = result.word if result else None
        confidence   = result.confidence if result else 0.0

        # In Typing Mode — build sentence/word
        if mode == "Typing Mode":
            word_added = builder.feed(current_word)
            if word_added and current_word:
                # Alphabet mode: speak the letter; word mode: speak the word
                speech.speak_if_new(current_word)
        else:
            builder.feed(None)

        if mode == "Text + Voice" and current_word:
            speech.speak_if_new(current_word)

        hold_progress = builder.get_hold_progress() if mode == "Typing Mode" else 0.0
        sentence      = builder.sentence if mode == "Typing Mode" else ""
        mode_key      = {"Display Only": "display", "Text + Voice": "voice", "Typing Mode": "typing"}.get(mode, "display")

        if hands:
            for hand in hands:
                x1, y1, x2, y2 = detector.get_bounding_box(hand)
                color = (100, 255, 150) if current_word else (100, 150, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        h_f, w_f = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w_f, 70), (10, 14, 26), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Show current rec_mode on frame
        rec_label = "WORD MODE" if new_rec_mode == "words" else "ALPHABET MODE"
        rec_color = (100, 150, 255) if new_rec_mode == "words" else (50, 220, 180)
        cv2.putText(frame, rec_label, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, rec_color, 1, cv2.LINE_AA)

        if current_word:
            cv2.putText(frame, current_word, (8, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2, cv2.LINE_AA)

        if mode == "Typing Mode" and hold_progress > 0:
            bar_len   = int(w_f * hold_progress)
            bar_color = (50, 255, 130) if hold_progress < 1.0 else (50, 200, 255)
            cv2.rectangle(frame, (0, h_f - 8), (w_f, h_f), (20, 20, 30), -1)
            cv2.rectangle(frame, (0, h_f - 8), (bar_len, h_f), bar_color, -1)

        if sentence and mode == "Typing Mode":
            overlay2 = frame.copy()
            cv2.rectangle(overlay2, (0, h_f - 45), (w_f, h_f - 8), (10, 14, 26), -1)
            cv2.addWeighted(overlay2, 0.65, frame, 0.35, 0, frame)
            cv2.putText(frame, sentence[-60:], (10, h_f - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (163, 230, 53), 1, cv2.LINE_AA)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        st.session_state["last_word"]       = current_word or ""
        st.session_state["last_confidence"] = confidence
        st.session_state["sentence"]        = builder.sentence
        st.session_state["word_count"]      = builder.word_count
        st.session_state["frame_count"]     = frame_count
        st.session_state["hold_progress"]   = hold_progress

    cap.release()
    speech.shutdown()
    camera_placeholder.markdown("""
    <div class='camera-placeholder'>
        <span style='font-size:3rem;'>📷</span>
        <div style='font-weight:600; color:#64748b;'>Camera stopped</div>
    </div>""", unsafe_allow_html=True)


def _render_info_section():
    mode          = st.session_state.get("mode", "Display Only")
    rec_mode      = st.session_state.get("rec_mode", "words")
    last_word     = st.session_state.get("last_word", "")
    confidence    = st.session_state.get("last_confidence", 0.0)
    sentence      = st.session_state.get("sentence", "")
    word_count    = st.session_state.get("word_count", 0)
    frame_count   = st.session_state.get("frame_count", 0)
    camera_running= st.session_state.get("camera_running", False)
    hold_progress = st.session_state.get("hold_progress", 0.0)

    emoji_map = {
        "Hello":"👋","Yes":"👍","No":"✊","Thank You":"🙏",
        "Peace":"✌️","Namaste":"🙏","Please":"🤟","Sorry":"🙇",
        "OK":"👌","Bad":"👎","Stop":"✋","I Love You":"🫶",
    }
    # For alphabet letters just show the letter big
    if rec_mode == "alphabet" and last_word and len(last_word) == 1:
        word_emoji = "🔤"
    else:
        word_emoji = emoji_map.get(last_word, "🤚") if last_word else "👁️"

    display_word = last_word if last_word else ("Waiting..." if camera_running else "—")

    show_conf  = st.session_state.get("show_confidence", True)
    conf_pct   = int(confidence * 100)
    conf_color = "#6ee7f7" if confidence > 0.8 else "#f59e0b" if confidence > 0.5 else "#94a3b8"

    # Mode badge color
    badge_color = "#06b6d4" if rec_mode == "alphabet" else "#6366f1"
    badge_label = "🔤 ALPHABET MODE" if rec_mode == "alphabet" else "💬 WORD MODE"

    st.markdown(f"""
    <div style='text-align:center; margin-bottom:0.5rem;'>
        <span style='background:{badge_color}22; border:1px solid {badge_color}; border-radius:20px;
        padding:0.3rem 1rem; color:{badge_color}; font-size:0.78rem; font-weight:700;
        letter-spacing:0.08em;'>{badge_label}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='detection-card'>
        <div class='detection-label'>Detected Gesture</div>
        <div class='detection-word'>{display_word}</div>
        <div class='detection-emoji'>{word_emoji if last_word else ""}</div>
        {"" if not show_conf else f"<div class='conf-bar-bg'><div class='conf-bar-fill' style='width:{conf_pct}%; background:linear-gradient(90deg,{conf_color},#a78bfa);'></div></div><div style='font-size:0.72rem;color:{conf_color};'>{conf_pct}% confidence</div>"}
    </div>
    """, unsafe_allow_html=True)

    # ── Typing area — always visible when Typing Mode ──
    if mode == "Typing Mode":
        box_class = "typing-box-alpha" if rec_mode == "alphabet" else "typing-box-words"
        label_color = "#06b6d4" if rec_mode == "alphabet" else "#6366f1"
        label_text  = "🔤 Typed Letters" if rec_mode == "alphabet" else "💬 Typed Words"

        st.markdown(f"<div class='typing-label' style='color:{label_color};'>{label_text}</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='typing-box {box_class}'>
            {sentence if sentence else '<span style="color:#374151;">Start signing to type...</span>'}
        </div>
        """, unsafe_allow_html=True)

        if hold_progress > 0 and camera_running:
            prog_pct = int(hold_progress * 100)
            st.markdown(f"<div style='color:#64748b; font-size:0.75rem; margin-top:0.3rem;'>⏳ Hold gesture: {prog_pct}% confirmed</div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state["sentence"]   = ""
                st.session_state["word_count"] = 0
        with col2:
            if st.button("⎵ Space", use_container_width=True):
                st.session_state["sentence"] = st.session_state.get("sentence", "") + " "
        with col3:
            if st.button("🔊 Speak", use_container_width=True):
                st.code(sentence)

    else:
        # Show mode info when not in Typing Mode
        mode_info = {
            "Display Only": ("👁️", "#3b82f6", "Screen only — no audio"),
            "Text + Voice": ("🔊", "#8b5cf6", "Screen + automatic speech"),
        }
        ico, clr, desc = mode_info.get(mode, ("▪", "#64748b", ""))
        st.markdown(f"""
        <div style='background:#1a2035; border:1px solid #2d3748; border-left:3px solid {clr};
        border-radius:10px; padding:0.75rem 1rem; margin:0.5rem 0;'>
            <span style='font-size:1rem;'>{ico}</span>
            <span style='color:#e2e8f0; font-weight:600; font-size:0.88rem; margin-left:0.5rem;'>{mode}</span>
            <div style='color:#64748b; font-size:0.75rem; margin-top:0.2rem;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Stats ──
    st.markdown("<hr style='border-color:#1e2433; margin: 1rem 0 0.75rem;'>", unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    with s1:
        count_label = "Letters" if rec_mode == "alphabet" else "Words"
        st.markdown(f"<div class='stat-card'><div class='stat-value'>{word_count}</div><div class='stat-label'>{count_label}</div></div>", unsafe_allow_html=True)
    with s2:
        runtime = int(time.time() - st.session_state.get("session_start", time.time()))
        mins, secs = divmod(runtime, 60)
        st.markdown(f"<div class='stat-card'><div class='stat-value'>{mins}:{secs:02d}</div><div class='stat-label'>Session</div></div>", unsafe_allow_html=True)
    with s3:
        fps = min(30, frame_count // max(1, runtime)) if camera_running else 0
        st.markdown(f"<div class='stat-card'><div class='stat-value'>{fps}</div><div class='stat-label'>FPS</div></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-pill'>🎓 Final Year Project · Sign Language Translator · Computer Vision + NLP</div>
    """, unsafe_allow_html=True)
