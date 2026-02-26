"""
Multi-Layer Face Anti-Spoofing Engine (DeepFAS-inspired)
=========================================================
Design follows the taxonomy of "Deep Learning for Face Anti-Spoofing: A Survey"
(TPAMI 2022): https://github.com/ZitongYu/DeepFAS

Combines hybrid (handcrafted) cues + temporal (motion/blink) to detect
print, replay, and screen attacks. Each layer scores 0.0–1.0 (1.0 = live).

Static layers (single frame)
----------------------------
1. LBP Texture        – Real skin has rich micro-texture; flat media does not.
2. Moiré / FFT        – Screens emit periodic grid patterns (frequency domain).
3. Color Distribution  – Real skin: warm HSV, broad hue spread; screens flatter.
4. Edge Density        – 3D faces yield strong edges; printed photos softer.
5. Specular Highlights – Live faces: specular spots; flat media rarely.
6. Central Difference – CDCN-inspired (CVPR'20): gradient structure; live skin
   has richer central-difference response than flat prints/screens.
   Ref: https://github.com/ZitongYu/CDCN

Temporal (multi-frame)
----------------------
7. Motion  – Frame-to-frame variance (static image → spoof).
8. Blink   – Eye Aspect Ratio; no blink in sequence → likely photo/video.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# ─── Optional imports ────────────────────────────────────────────────────────
try:
    from skimage.feature import local_binary_pattern
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    import mediapipe as mp
    MEDIAPIPE_OK = True
except (ImportError, TypeError, Exception):
    MEDIAPIPE_OK = False
    mp = None


# ═══════════════════════════════════════════════════════════════════════════════
# Tunable thresholds / weights
# ═══════════════════════════════════════════════════════════════════════════════
COMPOSITE_THRESHOLD = 0.60          # below this → spoof (stricter: block images/screens)

WEIGHTS = {
    "lbp":       0.20,
    "moire":     0.20,
    "color":     0.18,
    "edge":      0.12,
    "specular":  0.10,
    "cdc":       0.20,   # Central Difference (CDCN-inspired)
}

# Per-layer knobs
LBP_RADIUS          = 1
LBP_N_POINTS        = 8
LBP_VAR_LIVE_MIN    = 0.0025        # higher bar (photos are flatter)

MOIRE_HIGH_RATIO_MAX = 0.32         # stricter for screens         # high-freq energy ratio above this → likely screen

COLOR_SAT_LIVE_MIN   = 35.0         # real skin has more saturation
COLOR_HUE_STD_MIN    = 14.0         # more hue spread for live skin

EDGE_RATIO_LIVE_MIN  = 0.05         # printed photos often softer
EDGE_RATIO_MAX       = 0.28

SPECULAR_BRIGHT_THRES = 228
SPECULAR_RATIO_MIN   = 0.0025

# Central Difference (CDCN-inspired): gradient structure variance
CDC_VAR_LIVE_MIN = 8.0    # below this → flat → spoof (tuned for 64x64 diff map)

# Sequence: motion and blink
MOTION_VAR_MIN = 2.5e-5             # frame-to-frame variance below this → static → spoof
MIN_FRAMES_FOR_MOTION = 3
EAR_BLINK_THRESHOLD = 0.22          # EAR below this = blink
EAR_MIN_FRAMES = 4
BLINK_REQUIRED = True               # require at least one blink in sequence


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        return (img * 255).clip(0, 255).astype(np.uint8)
    return img


def _to_gray(img: np.ndarray) -> np.ndarray:
    img = _to_uint8(img)
    if img.ndim == 3:
        if CV2_OK:
            code = cv2.COLOR_RGBA2GRAY if img.shape[2] == 4 else cv2.COLOR_RGB2GRAY
            return cv2.cvtColor(img, code)
        return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.uint8)
    return img


def _to_hsv(img: np.ndarray) -> np.ndarray:
    img = _to_uint8(img)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[2] == 4:
        img = img[..., :3]
    if CV2_OK:
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Minimal fallback – enough for heuristic scoring
    r, g, b = img[..., 0].astype(float), img[..., 1].astype(float), img[..., 2].astype(float)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn + 1e-10
    h = np.where(mx == r, 60 * ((g - b) / diff) % 360,
        np.where(mx == g, 60 * ((b - r) / diff) + 120,
                           60 * ((r - g) / diff) + 240))
    s = np.where(mx == 0, 0, (diff / (mx + 1e-10)) * 255)
    v = mx
    return np.stack([h / 2, s, v], axis=-1).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# Individual scoring layers (each returns 0.0 – 1.0, higher = more live-like)
# ═══════════════════════════════════════════════════════════════════════════════

def _score_lbp(gray: np.ndarray) -> float:
    """LBP histogram variance — rich texture ⇒ high score."""
    if not SKIMAGE_OK:
        return 0.5  # neutral fallback
    lbp = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, method="uniform")
    n_bins = LBP_N_POINTS + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    var = float(np.var(hist))
    # Map variance to 0-1. Anything ≥ 2× the threshold is fully live.
    score = min(1.0, var / (LBP_VAR_LIVE_MIN * 2))
    return score


def _score_moire(gray: np.ndarray) -> float:
    """
    FFT high-frequency energy ratio.
    Screens produce periodic moiré patterns that concentrate energy at
    specific high frequencies.  A high ratio → likely screen → low score.
    """
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    # Define "low frequency" as the central 30% of the spectrum
    r = int(min(rows, cols) * 0.15)
    mask_low = np.zeros_like(magnitude, dtype=bool)
    y, x = np.ogrid[:rows, :cols]
    mask_low[((y - crow)**2 + (x - ccol)**2) <= r**2] = True

    total = magnitude.sum() + 1e-10
    low_energy = magnitude[mask_low].sum()
    high_ratio = 1.0 - (low_energy / total)

    # high_ratio close to 1 means most energy is high-freq → moiré likely
    if high_ratio >= MOIRE_HIGH_RATIO_MAX:
        score = max(0.0, 1.0 - (high_ratio - MOIRE_HIGH_RATIO_MAX) / 0.3)
    else:
        score = 1.0
    return float(score)


def _score_color(hsv: np.ndarray) -> float:
    """
    HSV colour analysis.
    Real skin has warm hue, moderate-to-high saturation, and broad hue spread.
    Screen reproductions tend to have shifted hue and flat saturation.
    """
    h, s, v = hsv[..., 0].astype(float), hsv[..., 1].astype(float), hsv[..., 2].astype(float)

    mean_sat = float(np.mean(s))
    hue_std  = float(np.std(h))

    sat_score = min(1.0, mean_sat / (COLOR_SAT_LIVE_MIN * 2.0))
    hue_score = min(1.0, hue_std / (COLOR_HUE_STD_MIN * 2.0))

    return 0.5 * sat_score + 0.5 * hue_score


def _score_edge(gray: np.ndarray) -> float:
    """
    Canny edge density.
    3-D faces yield strong depth/shadow edges; printed photos are softer.
    """
    if not CV2_OK:
        return 0.5
    edges = cv2.Canny(gray, 50, 150)
    ratio = float(np.count_nonzero(edges)) / max(edges.size, 1)
    ratio = min(ratio, EDGE_RATIO_MAX)
    score = min(1.0, ratio / (EDGE_RATIO_LIVE_MIN * 2.0))
    return score


def _score_specular(hsv: np.ndarray) -> float:
    """
    Specular highlight detection.
    Real 3D faces reflect light → bright spots on nose / forehead.
    Flat media rarely reproduces these.
    """
    v = hsv[..., 2]
    bright = np.count_nonzero(v >= SPECULAR_BRIGHT_THRES)
    total = max(v.size, 1)
    ratio = bright / total
    score = min(1.0, ratio / (SPECULAR_RATIO_MIN * 3.0))
    return float(score)


def _score_central_difference(gray: np.ndarray) -> float:
    """
    Central-difference (CDCN-inspired) cue: gradient structure.
    CDCN (CVPR'20) uses central difference convolution to capture fine-grained
    structure; live skin has richer local gradient variance than flat prints.
    We approximate with Laplacian response variance on the face crop.
    Ref: https://github.com/ZitongYu/CDCN
    """
    if gray.size < 100:
        return 0.5
    g = _to_uint8(gray).astype(np.float32)
    if CV2_OK:
        # Laplacian: center-weighted difference from neighbors (CDCN-like)
        lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    else:
        # 3x3 Laplacian via numpy: center - (L+R+U+D)
        h, w = g.shape
        c = g[1:-1, 1:-1]
        lap = 4.0 * c - (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
        lap = np.pad(lap, 1, mode="edge").astype(np.float32)
    var = float(np.var(lap))
    score = min(1.0, var / (CDC_VAR_LIVE_MIN * 4.0)) if CDC_VAR_LIVE_MIN else 1.0
    return score


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def check_liveness(face_array: np.ndarray) -> dict:
    """
    Parameters
    ----------
    face_array : np.ndarray
        Cropped face region (RGB, uint8 or float32, any resolution).

    Returns
    -------
    dict
        is_live  : bool
        score    : float   (composite 0-1, higher = more live)
        scores   : dict    (per-layer breakdown)
        reason   : str     (human-readable reason if spoof)
        method   : str
    """
    if face_array is None or face_array.size == 0:
        return {
            "is_live": False, "score": 0.0,
            "scores": {}, "reason": "Empty face input", "method": "empty",
        }

    gray = _to_gray(face_array)
    hsv  = _to_hsv(face_array)

    # Run all layers (including CDCN-inspired central difference)
    layer_scores = {
        "lbp":      _score_lbp(gray),
        "moire":    _score_moire(gray),
        "color":    _score_color(hsv),
        "edge":     _score_edge(gray),
        "specular": _score_specular(hsv),
        "cdc":      _score_central_difference(gray),
    }

    # Weighted composite
    composite = sum(WEIGHTS[k] * layer_scores[k] for k in WEIGHTS)
    composite = round(composite, 4)

    is_live = composite >= COMPOSITE_THRESHOLD

    # Determine the weakest signal for the reason string
    reason = ""
    if not is_live:
        weakest = min(layer_scores, key=lambda k: layer_scores[k])
        reason_map = {
            "lbp":      "Flat texture — possible printed photo",
            "moire":    "Screen moiré pattern — possible video / phone replay",
            "color":    "Abnormal colour — possible screen reproduction",
            "edge":     "Low edge detail — possible printed photo",
            "specular": "No specular highlights — possible flat surface",
            "cdc":      "Flat gradient structure — possible photo or screen (CDCN cue)",
        }
        reason = reason_map.get(weakest, "Liveness check failed")

    logger.info(
        f"[AntiSpoof] composite={composite:.3f} live={is_live} "
        f"layers={{{', '.join(f'{k}={v:.3f}' for k, v in layer_scores.items())}}}"
    )

    return {
        "is_live": is_live,
        "score":   composite,
        "scores":  {k: round(v, 4) for k, v in layer_scores.items()},
        "reason":  reason,
        "method":  "multi_layer_v1",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Motion and blink (sequence liveness)
# ═══════════════════════════════════════════════════════════════════════════════

def _motion_score(face_arrays: list) -> float:
    """
    Frame-to-frame variance in face region. Static image → near-zero variance → 0.
    Returns 0.0–1.0 (1.0 = enough motion).
    """
    if not face_arrays or len(face_arrays) < MIN_FRAMES_FOR_MOTION:
        return 0.5  # neutral if too few frames
    grays = []
    for arr in face_arrays:
        if arr is None or arr.size == 0:
            continue
        g = _to_gray(arr)
        if g.size < 100:
            continue
        # Resize to fixed size for consistent variance
        if CV2_OK:
            g = cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA)
        else:
            from PIL import Image
            g = np.array(Image.fromarray(g).resize((64, 64), Image.Resampling.LANCZOS))
        grays.append(g.astype(np.float32))
    if len(grays) < 2:
        return 0.5
    variances = []
    for i in range(1, len(grays)):
        diff = np.abs(grays[i] - grays[i - 1])
        variances.append(float(np.mean(diff ** 2)))
    mean_var = np.mean(variances) if variances else 0.0
    score = min(1.0, mean_var / (MOTION_VAR_MIN * 10)) if MOTION_VAR_MIN else 1.0
    return float(score)


def _ear_from_landmarks(landmarks, idx1, idx2, idx3, idx4, idx5, idx6):
    """EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)."""
    p1 = np.array([landmarks[idx1].x, landmarks[idx1].y])
    p2 = np.array([landmarks[idx2].x, landmarks[idx2].y])
    p3 = np.array([landmarks[idx3].x, landmarks[idx3].y])
    p4 = np.array([landmarks[idx4].x, landmarks[idx4].y])
    p5 = np.array([landmarks[idx5].x, landmarks[idx5].y])
    p6 = np.array([landmarks[idx6].x, landmarks[idx6].y])
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h = 2 * np.linalg.norm(p1 - p4)
    if h < 1e-6:
        return 0.3
    return (v1 + v2) / h


# MediaPipe Face Mesh eye indices: left 33,133,160,158,153,144; right 362,263,385,387,373,380
_LEFT_EYE = (33, 133, 160, 158, 153, 144)
_RIGHT_EYE = (362, 263, 385, 387, 373, 380)

_face_mesh = None

def _get_face_mesh():
    global _face_mesh
    if _face_mesh is None and MEDIAPIPE_OK:
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
    return _face_mesh


def _blink_detected(face_arrays: list) -> tuple:
    """
    Returns (has_blink: bool, ear_scores: list). Uses EAR; below EAR_BLINK_THRESHOLD = blink.
    """
    if not MEDIAPIPE_OK or len(face_arrays) < EAR_MIN_FRAMES:
        return True, []  # no blink required if we can't check
    mesh = _get_face_mesh()
    if mesh is None:
        return True, []
    ear_scores = []
    for arr in face_arrays:
        if arr is None or arr.size == 0:
            continue
        img = _to_uint8(arr)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[2] == 4:
            img = img[..., :3]
        results = mesh.process(img)
        if not results.multi_face_landmarks:
            continue
        lm = results.multi_face_landmarks[0]
        ear_left = _ear_from_landmarks(lm.landmark, *_LEFT_EYE)
        ear_right = _ear_from_landmarks(lm.landmark, *_RIGHT_EYE)
        ear = (ear_left + ear_right) / 2.0
        ear_scores.append(ear)
    if len(ear_scores) < EAR_MIN_FRAMES:
        return True, ear_scores
    has_blink = any(e < EAR_BLINK_THRESHOLD for e in ear_scores)
    return has_blink, ear_scores


def check_liveness_sequence(face_arrays: list) -> dict:
    """
    Multi-frame liveness: single-frame composite + motion + blink.
    face_arrays: list of cropped face numpy arrays (RGB).
    Returns same shape as check_liveness; is_live False if any check fails.
    """
    if not face_arrays:
        return {
            "is_live": False, "score": 0.0,
            "scores": {}, "reason": "No frames", "method": "sequence",
        }
    # Single-frame checks on the latest frame
    latest = face_arrays[-1] if face_arrays else None
    single = check_liveness(latest) if latest is not None and latest.size > 0 else {
        "is_live": False, "score": 0.0, "scores": {}, "reason": "No face", "method": "single",
    }
    if not single["is_live"]:
        return single

    # Motion: require some frame-to-frame change (reject static photo)
    motion = _motion_score(face_arrays)
    if motion < 0.15:  # very low motion → likely static image
        logger.info(f"[AntiSpoof] sequence: motion too low ({motion:.4f}) → spoof")
        return {
            "is_live": False,
            "score": round(single["score"] * 0.5, 4),
            "scores": {**single.get("scores", {}), "motion": round(motion, 4)},
            "reason": "No motion detected — possible photo or screen.",
            "method": "sequence",
        }

    # Blink: require at least one blink in sequence (reject photo/video without blink)
    has_blink, ear_scores = _blink_detected(face_arrays)
    if BLINK_REQUIRED and len(ear_scores) >= EAR_MIN_FRAMES and not has_blink:
        logger.info(f"[AntiSpoof] sequence: no blink in {len(ear_scores)} frames → spoof")
        return {
            "is_live": False,
            "score": round(single["score"] * 0.6, 4),
            "scores": {**single.get("scores", {}), "blink": 0.0},
            "reason": "No blink detected — please look at the camera and blink naturally.",
            "method": "sequence",
        }

    return {
        "is_live": True,
        "score": single["score"],
        "scores": single.get("scores", {}),
        "reason": "",
        "method": "sequence",
    }
