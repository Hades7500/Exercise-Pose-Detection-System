import cv2
import numpy as np
from typing import Optional



# COLOURS
COL_WHITE      = (255, 255, 255)
COL_BLACK      = (0,   0,   0)
COL_GOOD       = (90,  210, 0)      # green
COL_WARN       = (0,   165, 255)    # orange
COL_BAD        = (0,   80,  220)    # red
COL_PANEL_BG   = (20,  20,  20)     # near-black panel
COL_ACCENT     = (255, 200, 60)     # gold accent

def _alpha_rect(frame: np.ndarray,
                x: int, y: int, w: int, h: int,
                colour=(20, 20, 20), alpha: float = 0.55,
                radius: int = 12):
    """Draw a semi-transparent rounded rectangle."""
    overlay = frame.copy()
    # Top-left, bottom-right corners with clamping
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
    cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _text_shadow(frame, text, pos, font, scale, colour, thickness=1):
    """Draw text with a subtle shadow for readability."""
    x, y = pos
    # Shadow
    cv2.putText(frame, text, (x + 1, y + 1),
                font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    # Main text
    cv2.putText(frame, text, pos,
                font, scale, colour, thickness, cv2.LINE_AA)


def _get_pixel(landmarks, name_idx: int, h: int, w: int):
    """Convert a landmark's normalised coords to pixel coords."""
    lm = landmarks[name_idx]
    return int(lm.x * w), int(lm.y * h)


# ─────────────────────────────────────────────
# ANGLE LABEL on joint
# ─────────────────────────────────────────────

def draw_angle_at_joint(frame: np.ndarray,
                        landmarks,
                        joint_idx: int,
                        angle: float,
                        good: bool = True):
    """Draw angle value next to a joint point on the frame."""
    h, w = frame.shape[:2]
    cx, cy = _get_pixel(landmarks, joint_idx, h, w)
    colour = COL_GOOD if good else COL_WARN
    label  = f"{int(angle)}\u00b0"

    # Small background pill
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    pad = 4
    cv2.rectangle(frame,
                  (cx + 8 - pad, cy - th - pad),
                  (cx + 8 + tw + pad, cy + pad),
                  COL_PANEL_BG, -1)

    cv2.putText(frame, label,
                (cx + 8, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                colour, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────
# MAIN OVERLAY FUNCTION
# ─────────────────────────────────────────────

def draw_coach_overlay(frame: np.ndarray,
                       result: dict,
                       landmarks=None,
                       show_angles: bool = True) -> np.ndarray:
    """
    Draw the full coaching UI onto the video frame.

    Args:
        frame:        BGR OpenCV frame (modified in-place)
        result:       dict returned by FormCoach.update()
        landmarks:    MediaPipe landmark list (for angle annotations)
        show_angles:  whether to annotate joint angles on the skeleton

    Returns:
        The annotated frame (same object as input).
    """
    h, w = frame.shape[:2]
    font  = cv2.FONT_HERSHEY_SIMPLEX
    label = result.get("label", "good_form")
    advice = result.get("advice", "")
    reps   = result.get("reps", 0)
    conf   = result.get("confidence", 0.0)
    ex     = result.get("exercise", "")
    feats  = result.get("features", {})

    # ── 1. Top-left exercise + rep counter panel ──────────────
    panel_w, panel_h = 220, 80
    _alpha_rect(frame, 10, 10, panel_w, panel_h)

    ex_display = ex.replace("_", " ").title() if ex else "No exercise"
    _text_shadow(frame, ex_display,
                 (20, 38), font, 0.55, COL_ACCENT, 1)

    _text_shadow(frame, "Reps",
                 (20, 62), font, 0.45, (180, 180, 180), 1)
    _text_shadow(frame, str(reps),
                 (70, 62), font, 0.72, COL_WHITE, 2)

    # ── 2. Bottom advice banner ───────────────────────────────
    advice_colour = result.get("colour", COL_GOOD)

    banner_h = 52
    _alpha_rect(frame, 0, h - banner_h, w, banner_h,
                colour=COL_PANEL_BG, alpha=0.72)

    # Coloured left accent bar
    cv2.rectangle(frame,
                  (0, h - banner_h),
                  (5, h),
                  advice_colour, -1)

    # Advice text — scale down if too long
    text_scale = 0.60 if len(advice) < 40 else 0.50
    _text_shadow(frame, advice,
                 (16, h - 18),
                 font, text_scale, COL_WHITE, 1)

    # ── 3. Confidence bar (bottom-right) ─────────────────────
    bar_w, bar_h = 120, 8
    bar_x = w - bar_w - 14
    bar_y = h - banner_h + 20

    # Background track
    cv2.rectangle(frame,
                  (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h),
                  (60, 60, 60), -1)

    # Fill
    fill = int(bar_w * conf)
    bar_colour = COL_GOOD if conf > 0.75 else (COL_WARN if conf > 0.5 else COL_BAD)
    if fill > 0:
        cv2.rectangle(frame,
                      (bar_x, bar_y),
                      (bar_x + fill, bar_y + bar_h),
                      bar_colour, -1)

    _text_shadow(frame, f"conf {conf:.0%}",
                 (bar_x, bar_y - 4),
                 font, 0.38, (160, 160, 160), 1)

    # ── 4. Joint angle annotations ────────────────────────────
    if show_angles and landmarks and feats:
        good = (label == "good_form")

        # Squat — knee angles
        if ex == "squat":
            draw_angle_at_joint(frame, landmarks, 25,
                                feats.get("left_knee_angle", 0), good)
            draw_angle_at_joint(frame, landmarks, 26,
                                feats.get("right_knee_angle", 0), good)

        # Pushup — hip angles
        elif ex == "pushup":
            draw_angle_at_joint(frame, landmarks, 23,
                                feats.get("left_hip_angle", 0), good)
            draw_angle_at_joint(frame, landmarks, 24,
                                feats.get("right_hip_angle", 0), good)
            draw_angle_at_joint(frame, landmarks, 13,
                                feats.get("left_elbow_angle", 0), good)
            draw_angle_at_joint(frame, landmarks, 14,
                                feats.get("right_elbow_angle", 0), good)

    return frame