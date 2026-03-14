"""
Speech Output Module
Handles text-to-speech using pyttsx3 (offline, no internet required).
"""

import threading
import queue
import time
from typing import Optional


class SpeechEngine:
    def __init__(self, rate: int = 130, volume: float = 1.0, voice_index: int = 0):
        # Slower, clearer speech rate for better understanding
        self.rate = rate
        self.volume = volume
        self.voice_index = voice_index
        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def speak(self, text: str):
        # Speak phrases more naturally, split on punctuation
        import re
        phrases = re.split(r'[,.!?;]+', text.strip())
        for phrase in phrases:
            phrase = phrase.strip()
            if phrase:
                while not self._queue.empty():
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        break
                self._queue.put(phrase)

    def shutdown(self):
        self._stop_event.set()
        self._queue.put(None)

    def _worker(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', self.rate)
            engine.setProperty('volume', self.volume)
            voices = engine.getProperty('voices')
            if voices and self.voice_index < len(voices):
                engine.setProperty('voice', voices[self.voice_index].id)
            while not self._stop_event.is_set():
                try:
                    text = self._queue.get(timeout=0.5)
                    if text is None:
                        break
                    engine.say(text)
                    engine.runAndWait()
                except queue.Empty:
                    continue
            engine.stop()
        except Exception as e:
            print(f"[SpeechEngine] Error: {e}")


class SpeechController:
    def __init__(self, cooldown_seconds: float = 2.0):
        self.engine = SpeechEngine()
        self.cooldown = cooldown_seconds
        self._last_word: Optional[str] = None
        self._last_spoken_at: float = 0.0
        self.enabled: bool = True

    def speak_if_new(self, word: str):
        if not self.enabled or not word:
            return
        now = time.time()
        is_new_word = word != self._last_word
        cooldown_passed = (now - self._last_spoken_at) >= self.cooldown
        if is_new_word or cooldown_passed:
            self.engine.speak(word)
            self._last_word = word
            self._last_spoken_at = now

    def speak(self, text: str):
        if self.enabled:
            self.engine.speak(text)

    def reset(self):
        self._last_word = None
        self._last_spoken_at = 0.0

    def shutdown(self):
        self.engine.shutdown()