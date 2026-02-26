# One Step Greener – Face Recognition Attendance

A web-based **face recognition attendance system** for waste management teams. Employees and field workers (manforce) check in and out using their face—no cards or PINs. The app includes **anti-spoofing** (liveness detection) to block photos, screens, and replay attacks.

---

## What it does

- **Register** users by capturing their face (with live guidance: position, size, centering). Supports **employees** (by employee ID) and **manforce** (by Aadhaar, name, mobile).
- **Attendance** punch in/out via webcam: first scan of the day = punch in, next = punch out. One-minute cooldown between punches.
- **Dashboard** shows today’s attendance (punch-in and punch-out times) and quick links to Attendance and Register.
- **Liveness checks** during registration and recognition to reject printed photos, phone screens, and video replays (texture, motion, blink, and other cues).

---

## Features

| Feature | Description |
|--------|-------------|
| **Face registration** | Multi-frame capture with real-time feedback (face detected, centered, big enough). Optional PIN to unlock the Register page. |
| **Face recognition** | Match live face to stored embeddings (FAISS + 512-d FaceNet). Returns name, punch type (in/out), timestamp. |
| **Anti-spoofing** | Multi-layer checks: LBP texture, Moiré/FFT, color, edges, specular, central-difference; plus motion and blink for sequences. |
| **User types** | **Employee**: ID + optional name. **Manforce**: Aadhaar, full name, mobile. |
| **Duplicate prevention** | Same face cannot be registered for two different people. |
| **Cooldown** | 1-minute cooldown per user between punches to avoid double taps. |
| **Today’s view** | Today’s attendance list with first punch-in and last punch-out per person. |

---

## Tech stack

- **Backend:** Flask (Python 3.10)
- **Face detection & embeddings:** MTCNN + InceptionResnetV1 (VGGFace2) via `facenet-pytorch`
- **Embedding search:** FAISS (L2 index, cosine similarity)
- **Anti-spoofing:** Custom pipeline (LBP, FFT/Moiré, color, edges, specular, CDCN-style; MediaPipe for blink)
- **Database:** SQLite (`employees`, `attendance` tables)
- **Frontend:** HTML/CSS/JS, camera capture via browser

---

## Project structure

```
.
├── app.py                 # Flask app, routes, API handlers
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker image for HF Spaces (port 7860)
├── database/
│   ├── db.py              # SQLite helpers (employees, attendance)
│   ├── constable.db       # SQLite DB (created at runtime)
│   ├── face_index.faiss   # FAISS index (created at runtime)
│   └── face_meta.json     # FAISS ID → employee_id mapping
├── models/
│   ├── face_engine.py     # MTCNN + InceptionResnetV1, decode/crop/embed
│   ├── embeddings_store.py # FAISS wrapper, add/search
│   └── anti_spoof.py      # Liveness (single frame + sequence)
├── static/
│   ├── css/style.css
│   ├── js/
│   │   ├── camera.js      # Shared camera logic
│   │   ├── register.js    # Registration flow + face-check
│   │   └── attendance.js  # Recognition + punch
│   └── images/
└── templates/
    ├── base.html
    ├── dashboard.html     # Home: Attendance + Register links
    ├── register.html      # Enroll employee / manforce
    └── attendance.html    # Punch in/out by face
```

---

## Prerequisites

- **Python 3.10** (or 3.8+)
- **Camera** for registration and attendance (browser will request access)
- **Optional:** GPU for faster face models (CUDA); runs on CPU otherwise

---

## Installation

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd hf-space
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# or: venv\Scripts\activate  # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

On Linux, OpenCV and other libs may need system packages:

```bash
# Debian/Ubuntu
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev
```

---

## Configuration

| Variable | Description | Default |
|----------|-------------|--------|
| `PORT` | HTTP port | `5000` (local) / `7860` (Docker/HF Spaces) |
| `SECRET_KEY` | Flask secret key | `constable-secret-2025` |
| `REGISTER_PIN` | PIN to unlock Register page | `3620` |
| `FLASK_DEBUG` | Set to `1` for debug mode | `0` |

Example:

```bash
export REGISTER_PIN=1234
export PORT=5000
```

---

## Running the app

### Local (development)

```bash
python app.py
```

Then open **http://localhost:5000** (or the port you set). You should see the dashboard with **Attendance** and **Register**.

### Docker (e.g. Hugging Face Spaces)

The Dockerfile is set up for **Hugging Face Spaces** (port **7860**):

```bash
docker build -t attendance-face .
docker run -p 7860:7860 attendance-face
```

Open **http://localhost:7860**.

---

## Usage instructions

### Dashboard (`/` or `/dashboard`)

- **Attendance** – Open the attendance page to punch in/out with your face.
- **Register** – Open the registration page (optionally enter a PIN if set).

### Register (`/register`)

1. Optionally enter the **Register PIN** (default `3620`) to unlock the form.
2. Choose **Employee** or **Manforce**:
   - **Employee:** Enter Employee ID (and optional name). Submit with face capture.
   - **Manforce:** Enter Aadhaar, full name, and mobile. Submit with face capture.
3. Allow camera access. Position your face in the oval; wait until the indicator shows **Ready** (face detected, centered, big enough).
4. Capture multiple frames when prompted. The app runs **liveness checks** (e.g. motion, blink); do not use a photo or screen.
5. On success, the person is stored in the DB and their face embeddings are added to the FAISS index. You can then use **Attendance** to punch in/out.

### Attendance (`/attendance`)

1. Open the Attendance page and allow camera access.
2. Look at the camera. The app will:
   - Detect your face and run **liveness** (single frame or sequence).
   - Match your face to the stored embeddings.
   - If matched: **first punch of the day** = punch **in**, **next** = punch **out** (with a 1-minute cooldown between punches).
3. You’ll see your name, punch type (in/out), and time. Today’s attendance is available from the dashboard.

### API (for integration)

| Endpoint | Method | Purpose |
|----------|--------|--------|
| `/api/face-check` | POST | Check if a frame has a valid face (centered, big enough). Body: `{ "frame": "<base64DataUrl>" }`. |
| `/api/register` | POST | Register employee or manforce. Body: `user_type`, `frames`, and either `employee_id` or `aadhaar`+`name`+`mobile`. |
| `/api/recognize` | POST | Recognize face and punch in/out. Body: `{ "frame": "..." }` or `{ "frames": ["...", ...] }`. |
| `/api/verify-pin` | POST | Verify Register PIN. Body: `{ "pin": "3620" }`. |
| `/api/employees` | GET | List all employees. |
| `/api/attendance/today` | GET | Today’s attendance records. |
| `/api/health` | GET | Health check + total indexed faces. |

---

## Notes

- **First run:** The app creates `database/constable.db`, `face_index.faiss`, and `face_meta.json` on first use. No manual DB setup required.
- **Hugging Face Spaces:** Use the Dockerfile and set the Space to use **Docker** and port **7860**.
- **Security:** Set `SECRET_KEY` and `REGISTER_PIN` in production; avoid default PIN in production.

---

## License

See repository license (if any).
