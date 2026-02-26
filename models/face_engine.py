"""
CONSTABLE – Face detection and recognition engine.
Uses:
  • MTCNN  – fast face detection & alignment
  • InceptionResnetV1 (pretrained='vggface2') – 512-d face embeddings
"""

import io
import base64
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ─── Lazy imports so the app starts even if GPU is not available ───────────
try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    import torch
    FACENET_OK = True
except ImportError:
    FACENET_OK = False
    logger.warning("facenet-pytorch not installed – face recognition disabled.")

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False


DEVICE = "cpu"
if FACENET_OK:
    try:
        import torch
        if torch.cuda.is_available():
            DEVICE = "cuda"
    except Exception:
        pass

_mtcnn = None
_resnet = None


def _get_models():
    global _mtcnn, _resnet
    if _mtcnn is None:
        _mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            keep_all=False,
            device=DEVICE,
        )
    if _resnet is None:
        _resnet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
    return _mtcnn, _resnet


# ─── Public API ────────────────────────────────────────────────────────────

def decode_image(data_url: str) -> Image.Image:
    """Convert a base64 data-URL to a PIL Image (RGB)."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return img


def check_face_in_frame(pil_image: Image.Image) -> dict:
    """
    Lightweight face check for live feedback (no embedding).
    Returns dict: face_detected, centered, big_enough.
    Face is centered if bbox center lies in middle 50% of image.
    Big enough if face width >= 80px and area >= 3% of image.
    """
    out = {"face_detected": False, "centered": False, "big_enough": False}
    if not FACENET_OK:
        return out
    mtcnn, _ = _get_models()
    try:
        boxes, _ = mtcnn.detect(pil_image)
    except Exception as e:
        logger.debug(f"Face check error: {e}")
        return out
    if boxes is None or len(boxes) == 0:
        return out
    w, h = pil_image.size
    b = boxes[0]
    x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    face_w = x2 - x1
    face_h = y2 - y1
    face_area = face_w * face_h
    img_area = w * h
    out["face_detected"] = True
    # Centered: face center in middle 50% of frame
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    out["centered"] = (0.25 * w <= cx <= 0.75 * w) and (0.25 * h <= cy <= 0.75 * h)
    # Big enough: width >= 80 and area >= 3% of image
    out["big_enough"] = face_w >= 80 and (face_area / max(img_area, 1)) >= 0.03
    return out


def get_face_embedding(pil_image: Image.Image):
    """
    Detect the largest face and return its 512-d embedding as a numpy array.
    Returns (embedding: np.ndarray, face_crop: np.ndarray) or (None, None).
    """
    if not FACENET_OK:
        return None, None

    mtcnn, resnet = _get_models()

    try:
        # MTCNN returns aligned face tensor (or None)
        face_tensor, prob = mtcnn(pil_image, return_prob=True)
    except Exception as e:
        logger.debug(f"MTCNN error: {e}")
        return None, None

    if face_tensor is None:
        return None, None

    # Get the face crop as numpy for anti-spoofing
    boxes, _ = mtcnn.detect(pil_image)
    face_crop = None
    if boxes is not None and len(boxes) > 0:
        b = boxes[0].astype(int)
        arr = np.array(pil_image)
        x1, y1, x2, y2 = max(0, b[0]), max(0, b[1]), b[2], b[3]
        face_crop = arr[y1:y2, x1:x2]

    import torch
    with torch.no_grad():
        embedding = resnet(face_tensor.unsqueeze(0).to(DEVICE))

    return embedding.squeeze().cpu().numpy(), face_crop


def get_embeddings_from_frames(data_urls: list):
    """
    Process a list of base64 frame data-URLs.
    Returns list of valid 512-d embeddings (may be empty).
    """
    embeddings = []
    for url in data_urls:
        try:
            img = decode_image(url)
            emb, _ = get_face_embedding(img)
            if emb is not None:
                embeddings.append(emb.tolist())
        except Exception as e:
            logger.debug(f"Frame processing error: {e}")
    return embeddings


def get_embeddings_and_crops_from_frames(data_urls: list):
    """
    Process a list of base64 frame data-URLs.
    Returns (embeddings: list of 512-d lists, face_crops: list of np.ndarray or None).
    face_crops[i] is the face crop for frame i (None if no face in that frame).
    """
    embeddings = []
    crops = []
    for url in data_urls:
        try:
            img = decode_image(url)
            emb, face_crop = get_face_embedding(img)
            if emb is not None:
                embeddings.append(emb.tolist())
                crops.append(face_crop)
            else:
                crops.append(None)
        except Exception as e:
            logger.debug(f"Frame processing error: {e}")
            crops.append(None)
    return embeddings, crops


def get_face_crops_from_frames(data_urls: list):
    """
    Get face crops only from a list of base64 frame data-URLs (for liveness sequence).
    Returns list of np.ndarray (face crops); frames with no face are omitted.
    """
    _, crops = get_embeddings_and_crops_from_frames(data_urls)
    return [c for c in crops if c is not None and c.size > 0]
