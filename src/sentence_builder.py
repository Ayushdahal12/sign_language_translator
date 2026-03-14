"""
Sentence Builder Module
Manages word collection, debouncing, and sentence construction.
"""

import time
from typing import List, Optional
from collections import deque


class SentenceBuilder:
    """
    Collects detected words into a sentence with debouncing to prevent
    the same word being added repeatedly from a held gesture.

    Features:
    - Debounce: same word only added after hold_time seconds
    - Confirmation: word added after it stays stable for confirm_frames
    - History: undo last word
    - Max length: prevent runaway sentences
    """

    def __init__(
        self,
        hold_time: float = 1.2,
        confirm_frames: int = 8,
        max_words: int = 30,
    ):
        self.hold_time = hold_time
        self.confirm_frames = confirm_frames
        self.max_words = max_words

        self._words: List[str] = []
        self._history: List[List[str]] = []   # for undo

        # Debounce state
        self._current_word: Optional[str] = None
        self._word_start_time: float = 0.0
        self._consecutive_frames: int = 0
        self._last_added_word: Optional[str] = None
        self._last_add_time: float = 0.0

    # ── Public API ──────────────────────────────

    @property
    def sentence(self) -> str:
        return " ".join(self._words)

    @property
    def words(self) -> List[str]:
        return list(self._words)

    @property
    def word_count(self) -> int:
        return len(self._words)

    def feed(self, word: Optional[str]) -> bool:
        """
        Feed the current detected word (or None if no gesture).
        Returns True if a new word was confirmed and added to the sentence.
        Provides visual feedback for user.
        """
        now = time.time()

        if word is None:
            self._current_word = None
            self._consecutive_frames = 0
            return False

        if word != self._current_word:
            self._current_word = word
            self._word_start_time = now
            self._consecutive_frames = 1
        else:
            self._consecutive_frames += 1

        # Check if gesture has been held long enough
        held_long_enough = (now - self._word_start_time) >= (self.hold_time * 0.85)  # slightly more responsive
        stable_enough    = self._consecutive_frames >= int(self.confirm_frames * 0.85)

        if held_long_enough and stable_enough:
            # Prevent adding the same word twice in quick succession
            same_as_last = (word == self._last_added_word)
            cooldown_ok  = (now - self._last_add_time) >= self.hold_time * 1.2

            if (not same_as_last or cooldown_ok) and len(self._words) < self.max_words:
                self._save_history()
                self._words.append(word)
                self._last_added_word = word
                self._last_add_time = now
                # Reset so it doesn't keep adding
                self._word_start_time = now + 999
                # Visual feedback: print or log (can be hooked to UI)
                print(f"[TypingMode] Added: {word}")
                return True

        return False

    def add_word(self, word: str):
        """Manually add a word (bypass debounce)."""
        if word and len(self._words) < self.max_words:
            self._save_history()
            self._words.append(word)

    def undo(self) -> bool:
        """Remove the last word. Returns True if successful."""
        if self._history:
            self._words = self._history.pop()
            return True
        if self._words:
            self._words.pop()
            return True
        return False

    def clear(self):
        """Clear the sentence."""
        if self._words:
            self._save_history()
        self._words = []
        self._current_word = None
        self._consecutive_frames = 0
        self._last_added_word = None

    def get_pending_word(self) -> Optional[str]:
        """Returns the word currently being held (not yet confirmed)."""
        return self._current_word

    def get_hold_progress(self) -> float:
        """Returns 0.0 – 1.0 progress toward confirming current gesture."""
        if self._current_word is None:
            return 0.0
        elapsed = time.time() - self._word_start_time
        return min(1.0, elapsed / self.hold_time)

    def _save_history(self):
        self._history.append(list(self._words))
        if len(self._history) > 20:
            self._history.pop(0)
