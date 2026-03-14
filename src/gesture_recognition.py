"""
Gesture Recognition Module
Two completely separate modes:
  - WORD MODE:     Hello, Yes, No, Thank You, Peace, Please, Sorry, OK, Namaste, Stop, Bad
  - ALPHABET MODE: A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
Rules based on real ASL hand positions.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict
from src.hand_detection import HandDetector, HandLandmarks


@dataclass
class GestureResult:
    word: str
    confidence: float
    emoji: str = ""
    description: str = ""


class GestureRecognizer:

    GESTURE_EMOJI: Dict[str, str] = {
        "Hello": "👋", "Yes": "👍", "No": "✊", "Thank You": "🙏",
        "Peace": "✌️", "Namaste": "🙏", "Please": "🤲", "Sorry": "🙇",
        "OK": "👌", "Bad": "👎", "Stop": "✋", "I Love You": "🫶",
        "A": "✊", "B": "🤚", "C": "🤏", "D": "☝️", "E": "✋",
        "F": "👌", "G": "👉", "H": "🤞", "I": "🤙", "J": "🤙",
        "K": "✌️", "L": "👆", "M": "✊", "N": "✊", "O": "👌",
        "P": "👇", "Q": "👇", "R": "🤞", "S": "✊", "T": "✊",
        "U": "✌️", "V": "✌️", "W": "🖖", "X": "☝️", "Y": "🤙", "Z": "☝️",
    }

    def __init__(self, mode: str = "words"):
        self.mode = mode
        self.detector = HandDetector(max_hands=2)

        self._word_rules = [
            ("i_love_you", self._rule_i_love_you),
            ("namaste",   self._rule_namaste),
            ("ok",        self._rule_ok),
            ("stop",      self._rule_stop),
            ("thank_you", self._rule_thank_you),
            ("peace",     self._rule_peace),
            ("please",    self._rule_please),
            ("sorry",     self._rule_sorry),
            ("hello",     self._rule_hello),
            ("yes",       self._rule_yes),
            ("bad",       self._rule_bad),
            ("no",        self._rule_no),
        ]

        self._alphabet_rules = [
            ("Y", self._rule_Y),
            ("I", self._rule_I),
            ("L", self._rule_L),
            ("F", self._rule_F),
            ("D", self._rule_D),
            ("K", self._rule_K),
            ("W", self._rule_W),
            ("H", self._rule_H),
            ("G", self._rule_G),
            ("U", self._rule_U),
            ("V", self._rule_V),
            ("R", self._rule_R),
            ("B", self._rule_B),
            ("C", self._rule_C),
            ("O", self._rule_O),
            ("E", self._rule_E),
            ("M", self._rule_M),
            ("N", self._rule_N),
            ("S", self._rule_S),
            ("T", self._rule_T),
            ("X", self._rule_X),
            ("A", self._rule_A),
        ]

    def set_mode(self, mode: str):
        self.mode = mode

    def recognize(self, hands: List[HandLandmarks]) -> Optional[GestureResult]:
        if not hands:
            return None
        rules = self._word_rules if self.mode == "words" else self._alphabet_rules
        for _, rule_fn in rules:
            result = rule_fn(hands)
            if result:
                word, confidence = result
                return GestureResult(
                    word=word,
                    confidence=confidence,
                    emoji=self.GESTURE_EMOJI.get(word, ""),
                )
        return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _dist(self, h, a, b):
        pa = np.array(h.landmarks[a][:2])
        pb = np.array(h.landmarks[b][:2])
        return float(np.linalg.norm(pa - pb))

    def _fingers(self, h):
        d = self.detector
        return (
            d.is_finger_up(h, 'thumb'),
            d.is_finger_up(h, 'index'),
            d.is_finger_up(h, 'middle'),
            d.is_finger_up(h, 'ring'),
            d.is_finger_up(h, 'pinky'),
        )

    def _lm(self, h, idx):
        return h.landmarks[idx]

    # ══════════════════════════════════════════════════════════════════════════
    # WORD RULES
    # ══════════════════════════════════════════════════════════════════════════

    def _rule_i_love_you(self, hands):
        """
        I Love You = Real ASL sign 🤟
        THUMB + INDEX + PINKY all extended up.
        MIDDLE and RING fingers folded down.
        Single hand only.
        """
        h = hands[0]
        thumb_up, index_up, middle_up, ring_up, pinky_up = self._fingers(h)
        if thumb_up and index_up and pinky_up and not middle_up and not ring_up:
            return ("Please", 0.90)
        return None

    def _rule_namaste(self, hands):
        if len(hands) < 2:
            return None
        w1 = np.array(hands[0].landmarks[HandDetector.WRIST][:2])
        w2 = np.array(hands[1].landmarks[HandDetector.WRIST][:2])
        if float(np.linalg.norm(w1 - w2)) < 0.25:
            return ("Namaste", 0.92)
        return None

    def _rule_ok(self, hands):
        h = hands[0]
        _, _, middle_up, ring_up, pinky_up = self._fingers(h)
        if self._dist(h, HandDetector.THUMB_TIP, HandDetector.INDEX_TIP) < 0.06 \
                and middle_up and ring_up and pinky_up:
            return ("OK", 0.88)
        return None

    def _rule_stop(self, hands):
        h = hands[0]
        if self.detector.count_fingers(h) == 5:
            if self._lm(h, HandDetector.WRIST)[2] > self._lm(h, HandDetector.MIDDLE_TIP)[2]:
                return ("Stop", 0.85)
        return None

    def _rule_thank_you(self, hands):
        h = hands[0]
        if self.detector.count_fingers(h) == 5:
            if self._lm(h, HandDetector.WRIST)[1] > self._lm(h, HandDetector.MIDDLE_MCP)[1]:
                return ("Thank You", 0.85)
        return None

    def _rule_peace(self, hands):
        h = hands[0]
        _, index_up, middle_up, ring_up, pinky_up = self._fingers(h)
        if index_up and middle_up and not ring_up and not pinky_up:
            return ("Peace", 0.90)
        return None

    def _rule_please(self, hands):
        h = hands[0]
        thumb_up, index_up, middle_up, ring_up, pinky_up = self._fingers(h)
        if index_up and middle_up and ring_up and pinky_up and not thumb_up:
            return ("I Love You", 0.87)
        return None

    def _rule_sorry(self, hands):
        h = hands[0]
        _, index_up, middle_up, ring_up, pinky_up = self._fingers(h)
        if index_up and middle_up and ring_up and not pinky_up:
            return ("Sorry", 0.85)
        return None

    def _rule_hello(self, hands):
        h = hands[0]
        _, index_up, middle_up, ring_up, pinky_up = self._fingers(h)
        if index_up and not middle_up and not ring_up and not pinky_up:
            return ("Hello", 0.90)
        return None

    def _rule_yes(self, hands):
        h = hands[0]
        thumb_up, index_up, middle_up, _, _ = self._fingers(h)
        if thumb_up and not index_up and not middle_up:
            if self._lm(h, HandDetector.THUMB_TIP)[1] < self._lm(h, HandDetector.WRIST)[1] - 0.1:
                return ("Yes", 0.88)
        return None

    def _rule_bad(self, hands):
        h = hands[0]
        thumb_up, index_up, middle_up, _, _ = self._fingers(h)
        if not thumb_up and not index_up and not middle_up:
            if self._lm(h, HandDetector.THUMB_TIP)[1] > self._lm(h, HandDetector.WRIST)[1] + 0.08:
                return ("Bad", 0.80)
        return None

    def _rule_no(self, hands):
        if self.detector.count_fingers(hands[0]) == 0:
            return ("No", 0.85)
        return None

    # ══════════════════════════════════════════════════════════════════════════
    # ALPHABET RULES — matched to real ASL image
    # ══════════════════════════════════════════════════════════════════════════

    def _rule_A(self, hands):
        """Fist, thumb resting on SIDE of index finger (not over, not under)."""
        h = hands[0]
        thumb_up, index_up, middle_up, ring_up, pinky_up = self._fingers(h)
        if not index_up and not middle_up and not ring_up and not pinky_up and not thumb_up:
            thumb_y = self._lm(h, HandDetector.THUMB_TIP)[1]
            index_y = self._lm(h, HandDetector.INDEX_TIP)[1]
            wrist_y = self._lm(h, HandDetector.WRIST)[1]
            # Thumb tip between index tip and wrist height
            if index_y < thumb_y < wrist_y:
                return ("A", 0.75)
        return None

    def _rule_B(self, hands):
        """4 fingers straight UP and TOGETHER, thumb folded across palm."""
        h = hands[0]
        thumb_up, index_up, middle_up, ring_up, pinky_up = self._fingers(h)
        if index_up and middle_up and ring_up and pinky_up and not thumb_up:
            idx_x   = self._lm(h, HandDetector.INDEX_TIP)[0]
            pinky_x = self._lm(h, HandDetector.PINKY_TIP)[0]
            if abs(idx_x - pinky_x) < 0.10:
                return ("B", 0.80)
        return None

    def _rule_C(self, hands):
        """Hand curved like C — thumb and fingers open with rounded gap."""
        h = hands[0]
        count = self.detector.count_fingers(h)
        if count in [1, 2, 3]:
            d_thumb_index = self._dist(h, HandDetector.THUMB_TIP, HandDetector.INDEX_TIP)
            d_thumb_pinky = self._dist(h, HandDetector.THUMB_TIP, HandDetector.PINKY_TIP)
            if 0.08 < d_thumb_index < 0.25 and d_thumb_pinky > 0.15:
                return ("C", 0.72)
        return None

    def _rule_D(self, hands):
        """Index finger UP, thumb touches middle finger, others curled."""
        h = hands[0]
        _, index_up, middle_up, ring_up, pinky_up = self._fingers(h)
        if index_up and not middle_up and not ring_up and not pinky_up:
            if self._dist(h, HandDetector.THUMB_TIP, HandDetector.MIDDLE_TIP) < 0.08:
                return ("D", 0.78)
        return None

    def _rule_E(self, hands):
        """All fingers bent/curled DOWN toward palm — like bent fist."""
        h = hands[0]
        thumb_up, index_up, middle_up, ring_up, pinky_up = self._fingers(h)
        if not index_up and not middle_up and not ring_up and not pinky_up and not thumb_up:
            idx_tip = self._lm(h, HandDetector.INDEX_TIP)[1]
            idx_mcp = self._lm(h, HandDetector.INDEX_MCP)[1]
            mid_tip = self._lm(h, HandDetector.MIDDLE_TIP)[1]
            mid_mcp = self._lm(h, HandDetector.MIDDLE_MCP)[1]
            if idx_tip > idx_mcp - 0.03 and mid_tip > mid_mcp - 0.03:
                thumb_y = self._lm(h, HandDetector.THUMB_TIP)[1]
                wrist_y = self._lm(h, HandDetector.WRIST)[1]
                if thumb_y < wrist_y + 0.05:
                    return ("E", 0.72)
        return None

    def _rule_F(self, hands):
        """Index+thumb circle, middle+ring+pinky pointing UP."""
        h = hands[0]
        _, _, middle_up, ring_up, pinky_up = self._fingers(h)
        if self._dist(h, HandDetector.THUMB_TIP, HandDetector.INDEX_TIP) < 0.07 \
                and middle_up and ring_up and pinky_up:
            return ("F", 0.80)
        return None

    def _rule_G(self, hands):
        """Index finger points SIDEWAYS horizontally, others folded."""
        h = hands[0]
        _, index_up, middle_up, ring_up, pinky_up = self._fingers(h)
        if index_up and not middle_up and not ring_up and not pinky_up:
            idx_tip = self._lm(h, HandDetector.INDEX_TIP)
            idx_mcp = self._lm(h, HandDetector.INDEX_MCP)
            dx = abs(idx_tip[0] - idx_mcp[0])
            dy = abs(idx_tip[1] - idx_mcp[1])
            if dx > dy * 1.2:
                return ("G", 0.72)
        return None

    def _rule_H(self, hands):
        """Index+middle both pointing SIDEWAYS horizontally together."""
        h = hands[0]
        _, index_up, middle_up, ring_up, pinky_up = self._fingers(h)
        if index_up and middle_up and not ring_up and not pinky_up:
            idx_tip = self._lm(h, HandDetector.INDEX_TIP)
            idx_mcp = self._lm(h, HandDetector.INDEX_MCP)
            dx = abs(idx_tip[0] - idx_mcp[0])
            dy = abs(idx_tip[1] - idx_mcp[1])
            if dx > dy * 1.2:
                return ("H", 0.72)
        return None

    def _rule_I(self, hands):
        """Only PINKY finger pointing UP, all others closed."""
        h = hands[0]
        _, index_up, middle_up, ring_up, pinky_up = self._fingers(h)
        if pinky_up and not index_up and not middle_up and not ring_up:
            return ("I", 0.85)
        return None

    def _rule_K(self, hands):
        """Index+middle V shape UP, thumb sticks out BETWEEN them."""
        h = hands[0]
        thumb_up, index_up, middle_up, ring_up, pinky_up = self._fingers(h)
        if index_up and middle_up and not ring_up and not pinky_up and thumb_up:
            tx = self._lm(h, HandDetector.THUMB_TIP)[0]
            ix = self._lm(h, HandDetector.INDEX_TIP)[0]
            mx = self._lm(h, HandDetector.MIDDLE_TIP)[0]
            if min(ix, mx) < tx < max(ix, mx):
                return ("K", 0.75)
        return None

    def _rule_L(self, hands):
        """Index UP + thumb OUT sideways making L shape."""
        h = hands[0]
        thumb_up, index_up, middle_up, ring_up, pinky_up = self._fingers(h)
        if index_up and thumb_up and not middle_up and not ring_up and not pinky_up:
            tx = self._lm(h, HandDetector.THUMB_TIP)[0]
            wx = self._lm(h, HandDetector.WRIST)[0]
            if abs(tx - wx) > 0.10:
                return ("L", 0.80)
        return None

    def _rule_M(self, hands):
        """THREE fingers folded DOWN over thumb. Thumb tucked under index+middle+ring."""
        h = hands[0]
        thumb_up, index_up, middle_up, ring_up, pinky_up = self._fingers(h)
        if not index_up and not middle_up and not ring_up and not pinky_up and not thumb_up:
            ty = self._lm(h, HandDetector.THUMB_TIP)[1]
            iy = self._lm(h, HandDetector.INDEX_TIP)[1]
            my = self._lm(h, HandDetector.MIDDLE_TIP)[1]
            ry = self._lm(h, HandDetector.RING_TIP)[1]
            if ty > iy and ty > my and ty > ry:
                return ("M", 0.70)
        return None

    def _rule_N(self, hands):
        """TWO fingers folded over thumb. Thumb between index+middle."""
        h = hands[0]
        thumb_up, index_up, middle_up, ring_up, pinky_up = self._fingers(h)
        if not index_up and not middle_up and not ring_up and not pinky_up and not thumb_up:
            tx = self._lm(h, HandDetector.THUMB_TIP)[0]
            ix = self._lm(h, HandDetector.INDEX_TIP)[0]
            mx = self._lm(h, HandDetector.MIDDLE_TIP)[0]
            ty = self._lm(h, HandDetector.THUMB_TIP)[1]
            iy = self._lm(h, HandDetector.INDEX_TIP)[1]
            my = self._lm(h, HandDetector.MIDDLE_TIP)[1]
            if min(ix, mx) < tx < max(ix, mx) and ty > iy and ty > my:
                return ("N", 0.70)
        return None

    def _rule_O(self, hands):
        """All fingers + thumb curved forming round O. All tips close together."""
        h = hands[0]
        t = np.array(self._lm(h, HandDetector.THUMB_TIP)[:2])
        i = np.array(self._lm(h, HandDetector.INDEX_TIP)[:2])
        m = np.array(self._lm(h, HandDetector.MIDDLE_TIP)[:2])
        r = np.array(self._lm(h, HandDetector.RING_TIP)[:2])
        if float(np.linalg.norm(t - i)) < 0.09 and \
           float(np.linalg.norm(t - m)) < 0.12 and \
           float(np.linalg.norm(t - r)) < 0.14:
            return ("O", 0.78)
        return None

    def _rule_R(self, hands):
        """Index+middle fingers CROSSED — both up but very close/overlapping."""
        h = hands[0]
        _, index_up, middle_up, ring_up, pinky_up = self._fingers(h)
        if index_up and middle_up and not ring_up and not pinky_up:
            if self._dist(h, HandDetector.INDEX_TIP, HandDetector.MIDDLE_TIP) < 0.04:
                return ("R", 0.74)
        return None

    def _rule_S(self, hands):
        """Fist with thumb wrapped OVER fingers — thumb tip above finger tips."""
        h = hands[0]
        thumb_up, index_up, middle_up, ring_up, pinky_up = self._fingers(h)
        if not index_up and not middle_up and not ring_up and not pinky_up and not thumb_up:
            ty = self._lm(h, HandDetector.THUMB_TIP)[1]
            iy = self._lm(h, HandDetector.INDEX_TIP)[1]
            my = self._lm(h, HandDetector.MIDDLE_TIP)[1]
            if ty < iy and ty < my:
                return ("S", 0.74)
        return None

    def _rule_T(self, hands):
        """Fist with thumb tucked BETWEEN index and middle fingers, pointing up."""
        h = hands[0]
        thumb_up, index_up, middle_up, ring_up, pinky_up = self._fingers(h)
        if not index_up and not middle_up and not ring_up and not pinky_up and thumb_up:
            tx  = self._lm(h, HandDetector.THUMB_TIP)[0]
            imx = self._lm(h, HandDetector.INDEX_MCP)[0]
            mmx = self._lm(h, HandDetector.MIDDLE_MCP)[0]
            if min(imx, mmx) < tx < max(imx, mmx):
                return ("T", 0.72)
        return None

    def _rule_U(self, hands):
        """Index+middle UP and held TOGETHER (close gap). Ring+pinky down."""
        h = hands[0]
        _, index_up, middle_up, ring_up, pinky_up = self._fingers(h)
        if index_up and middle_up and not ring_up and not pinky_up:
            if self._dist(h, HandDetector.INDEX_TIP, HandDetector.MIDDLE_TIP) < 0.05:
                return ("U", 0.78)
        return None

    def _rule_V(self, hands):
        """Index+middle UP and SPREAD apart in V shape. Ring+pinky down."""
        h = hands[0]
        _, index_up, middle_up, ring_up, pinky_up = self._fingers(h)
        if index_up and middle_up and not ring_up and not pinky_up:
            if self._dist(h, HandDetector.INDEX_TIP, HandDetector.MIDDLE_TIP) >= 0.05:
                return ("V", 0.78)
        return None

    def _rule_W(self, hands):
        """Index+middle+ring all UP and spread. Pinky and thumb down."""
        h = hands[0]
        _, index_up, middle_up, ring_up, pinky_up = self._fingers(h)
        if index_up and middle_up and ring_up and not pinky_up:
            return ("W", 0.80)
        return None

    def _rule_X(self, hands):
        """Index finger HOOKED/BENT — tip between PIP and MCP height."""
        h = hands[0]
        _, index_up, middle_up, ring_up, pinky_up = self._fingers(h)
        if not index_up and not middle_up and not ring_up and not pinky_up:
            idx_tip = self._lm(h, HandDetector.INDEX_TIP)[1]
            idx_pip = self._lm(h, 6)[1]
            idx_mcp = self._lm(h, HandDetector.INDEX_MCP)[1]
            if idx_mcp > idx_tip > idx_pip:
                return ("X", 0.70)
        return None

    def _rule_Y(self, hands):
        """THUMB + PINKY both out, other 3 fingers closed — shaka/hang loose."""
        h = hands[0]
        thumb_up, index_up, middle_up, ring_up, pinky_up = self._fingers(h)
        if thumb_up and pinky_up and not index_up and not middle_up and not ring_up:
            return ("Y", 0.85)
        return None
