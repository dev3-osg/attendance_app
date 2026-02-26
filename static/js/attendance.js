const video = document.getElementById('videoFeed');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const modal = document.getElementById('successModal');
const modalTitle = document.getElementById('modalTitle');
const modalName = document.getElementById('modalName');
const modalTime = document.getElementById('modalTime');
const cameraContainer = document.getElementById('cameraContainer');
const statusHint = document.getElementById('statusHint');

// Spoof toast
const spoofToast = document.getElementById('spoofToast');
const spoofToastMessage = document.getElementById('spoofToastMessage');

// Frame buffer for sequence liveness (motion + blink); need enough frames to catch a blink
const FRAME_BUFFER_SIZE = 14;
const CAPTURE_INTERVAL_MS = 280;
let frameBuffer = [];

let isScanning = true;
let stream = null;
let toastDismissTimer = null;
let lastFaceAlertAt = 0;
const FACE_ALERT_COOLDOWN_MS = 4000;

// ─── Camera ─────────────────────────────────────────────────────────────────

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'user', width: 640, height: 480 } 
        });
        video.srcObject = stream;
        startCaptureLoop();
    } catch (err) {
        console.error("Camera error:", err);
        statusText.textContent = "Camera access denied or unavailable";
        statusText.style.color = "red";
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
}

function captureFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    return canvas.toDataURL('image/jpeg', 0.8);
}

// ─── Capture loop ───────────────────────────────────────────────────────────

async function startCaptureLoop() {
    while (isScanning) {
        if (video.readyState === video.HAVE_ENOUGH_DATA) {
            const frame = captureFrame();
            frameBuffer.push(frame);
            if (frameBuffer.length > FRAME_BUFFER_SIZE) frameBuffer.shift();

            // Send sequence only when we have enough frames for blink detection (backend needs ~4+ frames)
const payload = frameBuffer.length >= 6
                ? { frames: frameBuffer.slice() }
                : { frame: frame };

            try {
                const response = await fetch('/api/recognize', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const result = await response.json();
                handleResult(result);
            } catch (e) {
                console.log("Network error", e);
            }
        }

        await new Promise(r => setTimeout(r, CAPTURE_INTERVAL_MS));
    }
}

// ─── Result handler ─────────────────────────────────────────────────────────

function handleResult(result) {
    if (result.status === 'success') {
        showSuccess(result);
        var action = (result.punch_type === 'out') ? 'Punched out' : 'Punched in';
        if (typeof showSnackbar === 'function') {
            showSnackbar(action + ' at ' + (result.timestamp || ''), 'success');
        }
    } else if (result.status === 'cooldown') {
        statusText.textContent = 'Please wait 1 min';
        statusText.style.color = "#FFD700";
        if (typeof showSnackbar === 'function') showSnackbar(result.message || 'Please wait 1 minute before punching again.', 'info');
    } else if (result.status === 'spoof') {
        showSpoofToast(result);
    } else if (result.status === 'unknown') {
        statusText.textContent = "Face not recognized";
        statusText.style.color = "#A5A5A5";
        if (typeof showSnackbar === 'function' && Date.now() - lastFaceAlertAt > FACE_ALERT_COOLDOWN_MS) {
            lastFaceAlertAt = Date.now();
            showSnackbar('Face not recognized — ensure your face is clearly visible.', 'info');
        }
    } else if (result.status === 'no_face') {
        statusText.textContent = "Position your face in the frame";
        statusText.style.color = "#A5A5A5";
        if (typeof showSnackbar === 'function' && Date.now() - lastFaceAlertAt > FACE_ALERT_COOLDOWN_MS) {
            lastFaceAlertAt = Date.now();
            showSnackbar('Adjust position — keep your face clearly visible in the frame.', 'info');
        }
    } else if (result.status === 'spoof' && result.reason && result.reason.toLowerCase().includes('blink')) {
        statusText.textContent = "Please blink to verify";
        statusText.style.color = "#FFD700";
    }
}

// ─── Spoof toast (banner, auto-dismiss) ─────────────────────────────────────

function showSpoofToast(data) {
    const msg = data.reason || data.message || "Use a live face, not a photo or screen.";
    spoofToastMessage.textContent = msg;
    spoofToast.classList.add('show');
    statusText.textContent = "⚠ Spoofing detected";
    statusText.style.color = "#FF3B30";

    if (toastDismissTimer) clearTimeout(toastDismissTimer);
    toastDismissTimer = setTimeout(() => {
        spoofToast.classList.remove('show');
        statusText.textContent = "Position your face in the frame";
        statusText.style.color = "#A5A5A5";
        toastDismissTimer = null;
    }, 4500);
}

// ─── Success modal ──────────────────────────────────────────────────────────

function showSuccess(data) {
    isScanning = false;
    statusDot.classList.add('active');
    if (statusHint) statusHint.style.visibility = 'hidden';

    if (modalTitle) modalTitle.textContent = (data.punch_type === 'out') ? 'Punched out' : 'Punched in';
    if (modalName) modalName.textContent = data.name;
    if (modalTime) modalTime.textContent = data.timestamp || '';
    
    modal.classList.add('show');
    
    // Auto dismiss after 4s
    setTimeout(dismissModal, 4000);
}

function dismissModal() {
    modal.classList.remove('show');
    statusDot.classList.remove('active');
    statusText.textContent = "Position your face in the frame";
    statusText.style.color = "#A5A5A5";
    if (statusHint) statusHint.style.visibility = "";
    isScanning = true;
    startCaptureLoop();
}

// ─── Init ───────────────────────────────────────────────────────────────────
startCamera();
window.addEventListener('beforeunload', stopCamera);
