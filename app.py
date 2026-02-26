"""
One Step Greener – Face recognition attendance (waste management).
Flask application entry point.
"""

import os
import logging
from flask import Flask, render_template, request, jsonify, redirect

from database.db import init_db, add_employee, get_employee, get_all_employees, mark_attendance, get_today_attendance
from models.embeddings_store import EmbeddingStore
from models.face_engine import (
    decode_image,
    get_face_embedding,
    check_face_in_frame,
    get_embeddings_from_frames,
    get_embeddings_and_crops_from_frames,
    get_face_crops_from_frames,
)
from models.anti_spoof import check_liveness, check_liveness_sequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "constable-secret-2025")

REGISTER_PIN = os.environ.get("REGISTER_PIN", "3620")

# ── Initialise database and embedding store ─────────────────────────────────
init_db()
store = EmbeddingStore()

# ══════════════════════════════════════════════════════════════════════════════
# Page routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/register")
def register_page():
    return render_template("register.html")


@app.route("/manage")
def manage_redirect():
    return redirect("/dashboard", code=302)


@app.route("/attendance")
def attendance_page():
    return render_template("attendance.html")


# ══════════════════════════════════════════════════════════════════════════════
# API routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/face-check", methods=["POST"])
def api_face_check():
    """
    Lightweight face-in-frame check for registration flow.
    Body: { frame: base64DataUrl }
    Returns: { face_detected, centered, big_enough, ready } (ready = all true).
    """
    data = request.get_json(force=True)
    frame = data.get("frame", "")
    if not frame:
        return jsonify({"face_detected": False, "centered": False, "big_enough": False, "ready": False})
    try:
        img = decode_image(frame)
    except Exception:
        return jsonify({"face_detected": False, "centered": False, "big_enough": False, "ready": False})
    r = check_face_in_frame(img)
    r["ready"] = r["face_detected"] and r["centered"] and r["big_enough"]
    return jsonify(r)


@app.route("/api/register", methods=["POST"])
def api_register():
    """
    Body JSON:
      Manforce: { user_type: 'manforce', aadhaar, name, mobile, frames }
      Employee: { user_type: 'employee', employee_id, frames }
    """
    data = request.get_json(force=True)
    user_type = (data.get("user_type") or "employee").strip().lower()
    frames = data.get("frames", [])

    if not frames:
        return jsonify({"status": "error", "message": "No frames provided."}), 400

    if user_type == "manforce":
        aadhaar = data.get("aadhaar", "").strip()
        name = data.get("name", "").strip()
        mobile = data.get("mobile", "").strip()
        if not aadhaar or not name or not mobile:
            return jsonify({"status": "error", "message": "Aadhaar number, full name and mobile number are required for Manforce."}), 400
        employee_id = aadhaar
    else:
        employee_id = data.get("employee_id", "").strip()
        if not employee_id:
            return jsonify({"status": "error", "message": "Employee code is required."}), 400
        name = data.get("name", "").strip() or employee_id
        aadhaar = ""
        mobile = ""

    logger.info(f"Registering {employee_id} ({name}, type={user_type}) with {len(frames)} frames …")
    embeddings, face_crops = get_embeddings_and_crops_from_frames(frames)

    if not embeddings:
        return jsonify({
            "status": "error",
            "message": "No face detected in the provided frames. "
                       "Please ensure good lighting and that your face is clearly visible."
        }), 400

    # Anti-spoofing: reject photo/screen/video (motion + blink + texture)
    if len(face_crops) >= 2:
        liveness = check_liveness_sequence([c for c in face_crops if c is not None and c.size > 0])
    else:
        liveness = check_liveness(face_crops[0]) if face_crops and face_crops[0] is not None else {"is_live": False}
    if not liveness.get("is_live", True):
        logger.warning(f"Registration rejected (spoof): {liveness.get('reason', 'liveness failed')}")
        return jsonify({
            "status": "spoof",
            "message": liveness.get("reason", "Liveness check failed. Use a live face, not a photo or screen."),
            "reason": liveness.get("reason", "Liveness check failed"),
            "composite": liveness.get("score", 0.0),
        }), 400

    # Check for duplicate face registration
    for emb in embeddings:
        match_id, score = store.search(emb)
        if match_id:
            match_emp = get_employee(match_id)
            match_name = match_emp["name"] if match_emp else match_id
            logger.warning(f"Registration rejected: face already registered to {match_name} ({match_id})")
            return jsonify({
                "status": "error",
                "message": f"This face is already registered to {match_name} ({match_id})."
            }), 400

    # Persist employee in DB and embeddings in FAISS
    add_employee(employee_id, name, user_type=user_type, aadhaar=aadhaar, mobile=mobile)
    store.add(employee_id, embeddings)

    logger.info(f"Registered {employee_id} with {len(embeddings)} embedding(s).")
    return jsonify({
        "status": "registered",
        "employee_id": employee_id,
        "name": name,
        "user_type": user_type,
        "embeddings_stored": len(embeddings),
    })


@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    """
    Body JSON:
      { frame: base64DataUrl }  or  { frames: [base64DataUrl, ...] }
    When frames is provided, uses sequence liveness (motion + blink).

    Response JSON (one of):
      { status: 'success',        name, timestamp }
      { status: 'already_marked', name }
      { status: 'spoof', reason, composite }
      { status: 'unknown' }
      { status: 'no_face' }
    """
    data = request.get_json(force=True)
    frame = data.get("frame", "")
    frames = data.get("frames", [])

    # Prefer frames for sequence liveness (motion + blink) when available
    if frames and len(frames) >= 2:
        try:
            face_crops = get_face_crops_from_frames(frames)
        except Exception:
            face_crops = []
        if not face_crops:
            return jsonify({"status": "no_face"})
        # Use latest frame for identity
        try:
            img = decode_image(frames[-1])
        except Exception:
            return jsonify({"status": "no_face"})
        embedding, _ = get_face_embedding(img)
        if embedding is None:
            return jsonify({"status": "no_face"})
        liveness = check_liveness_sequence(face_crops)
    else:
        if not frame:
            return jsonify({"status": "no_face"})
        try:
            img = decode_image(frame)
        except Exception:
            return jsonify({"status": "no_face"})
        embedding, face_crop = get_face_embedding(img)
        if embedding is None:
            return jsonify({"status": "no_face"})
        if face_crop is not None:
            liveness = check_liveness(face_crop)
        else:
            liveness = {"is_live": True}

    if not liveness.get("is_live", True):
        logger.info(f"Spoof detected (score={liveness.get('score', 0):.4f}, reason={liveness.get('reason', '')})")
        return jsonify({
            "status": "spoof",
            "reason": liveness.get("reason", "Liveness check failed"),
            "scores": liveness.get("scores", {}),
            "composite": liveness.get("score", 0.0),
        })

    # Identity search
    employee_id, score = store.search(embedding)
    if employee_id is None:
        return jsonify({"status": "unknown"})

    employee = get_employee(employee_id)
    name = employee["name"] if employee else employee_id

    result = mark_attendance(employee_id)

    if result["status"] == "cooldown":
        return jsonify({
            "status": "cooldown",
            "name": name,
            "message": "Please wait 1 minute before punching again.",
        })

    punch_type = result.get("punch_type", "in")
    logger.info(f"Punch {punch_type}: {employee_id} ({name}) at {result['timestamp']}")
    return jsonify({
        "status": "success",
        "name": name,
        "employee_id": employee_id,
        "timestamp": result["timestamp"],
        "punch_type": punch_type,
        "confidence": round(score, 4),
    })


@app.route("/api/verify-pin", methods=["POST"])
def api_verify_pin():
    """Verify PIN to unlock Register form for this page. PIN must match REGISTER_PIN (default 3620)."""
    data = request.get_json(force=True)
    pin = (data.get("pin") or "").strip()
    if pin == REGISTER_PIN:
        return jsonify({"status": "ok", "message": "Verified"})
    return jsonify({"status": "error", "message": "Incorrect PIN"}), 403


@app.route("/api/employees", methods=["GET"])
def api_employees_list():
    employees = get_all_employees()
    return jsonify({"status": "ok", "employees": employees, "count": len(employees)})


@app.route("/api/attendance/today", methods=["GET"])
def api_today_attendance():
    records = get_today_attendance()
    return jsonify({"status": "ok", "records": records, "count": len(records)})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "total_employees_indexed": store.total_vectors,
    })


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    logger.info(f"One Step Greener starting on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
