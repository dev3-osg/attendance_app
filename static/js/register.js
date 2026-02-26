const video = document.getElementById('videoFeed');
const progressCircle = document.getElementById('progressCircle');
const statusText = document.getElementById('statusText');

const tabEmployee = document.getElementById('tabEmployee');
const tabManforce = document.getElementById('tabManforce');
const panelEmployee = document.getElementById('panelEmployee');
const panelManforce = document.getElementById('panelManforce');

const manforceAadhaar = document.getElementById('manforceAadhaar');
const manforceName = document.getElementById('manforceName');
const manforceMobile = document.getElementById('manforceMobile');
const btnRegisterManforce = document.getElementById('btnRegisterManforce');
const statusTextManforce = document.getElementById('statusTextManforce');

const employeeCode = document.getElementById('employeeCode');
const btnRegisterEmployee = document.getElementById('btnRegisterEmployee');
const statusTextEmployee = document.getElementById('statusTextEmployee');

const spoofToast = document.getElementById('spoofToast');
const spoofToastMessage = document.getElementById('spoofToastMessage');

if (!video) console.warn('Register: video element missing');

let activeTab = 'employee';
let manforceFrames = [];
let employeeFrames = [];
let isCapturing = false;
let captureTarget = null;
const REQUIRED_FRAMES = 5;
const CAPTURE_INTERVAL_MS = 650;   // 5 frames over ~3.2s so blink is likely
const FACE_CHECK_POLL_MS = 500;
const MAX_RETRIES_PER_SLOT = 2;    // retry each slot up to 2 times if no face
let toastDismissTimer = null;
let faceCheckInterval = null;
let faceReady = false;

// Start camera
navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: 640, height: 480 } })
    .then(function(stream) { if (video) video.srcObject = stream; })
    .catch(function(err) { console.error(err); });

function switchTab(tab) {
    activeTab = tab;
    if (tab === 'employee') {
        if (tabEmployee) { tabEmployee.classList.add('active'); tabEmployee.setAttribute('aria-selected', 'true'); }
        if (tabManforce) { tabManforce.classList.remove('active'); tabManforce.setAttribute('aria-selected', 'false'); }
        if (panelEmployee) { panelEmployee.classList.add('active'); panelEmployee.removeAttribute('hidden'); }
        if (panelManforce) { panelManforce.classList.remove('active'); panelManforce.setAttribute('hidden', ''); }
    } else {
        if (tabManforce) { tabManforce.classList.add('active'); tabManforce.setAttribute('aria-selected', 'true'); }
        if (tabEmployee) { tabEmployee.classList.remove('active'); tabEmployee.setAttribute('aria-selected', 'false'); }
        if (panelManforce) { panelManforce.classList.add('active'); panelManforce.removeAttribute('hidden'); }
        if (panelEmployee) { panelEmployee.classList.remove('active'); panelEmployee.setAttribute('hidden', ''); }
    }
    updateVisibleButton();
}

function getFramesFor(type) {
    return type === 'manforce' ? manforceFrames : employeeFrames;
}

function setFramesFor(type, frames) {
    if (type === 'manforce') manforceFrames = frames; else employeeFrames = frames;
}

function updateVisibleButton() {
    if (activeTab === 'manforce') updateManforceButton(); else updateEmployeeButton();
}

function updateManforceButton() {
    if (!manforceAadhaar || !manforceName || !manforceMobile || !btnRegisterManforce) return;
    var valid = manforceAadhaar.value.trim().length > 0 && manforceName.value.trim().length > 0 && manforceMobile.value.trim().length > 0;
    if (isCapturing && captureTarget === 'manforce') {
        btnRegisterManforce.disabled = true;
        btnRegisterManforce.textContent = 'CAPTURING...';
    } else if (manforceFrames.length === REQUIRED_FRAMES) {
        btnRegisterManforce.disabled = false;
        btnRegisterManforce.textContent = 'Register';
        btnRegisterManforce.onclick = function() { submitRegistration('manforce'); };
    } else if (valid && faceReady) {
        btnRegisterManforce.disabled = false;
        btnRegisterManforce.textContent = 'START CAPTURE';
        btnRegisterManforce.onclick = function() { startCaptureProcess('manforce'); };
    } else if (valid) {
        btnRegisterManforce.disabled = true;
        btnRegisterManforce.textContent = 'START CAPTURE';
        btnRegisterManforce.onclick = null;
    } else {
        btnRegisterManforce.disabled = true;
        btnRegisterManforce.onclick = null;
    }
}

function updateEmployeeButton() {
    if (!employeeCode || !btnRegisterEmployee) return;
    var valid = employeeCode.value.trim().length > 0;
    if (isCapturing && captureTarget === 'employee') {
        btnRegisterEmployee.disabled = true;
        btnRegisterEmployee.textContent = 'CAPTURING...';
    } else if (employeeFrames.length === REQUIRED_FRAMES) {
        btnRegisterEmployee.disabled = false;
        btnRegisterEmployee.textContent = 'Register';
        btnRegisterEmployee.onclick = function() { submitRegistration('employee'); };
    } else if (valid && faceReady) {
        btnRegisterEmployee.disabled = false;
        btnRegisterEmployee.textContent = 'START CAPTURE';
        btnRegisterEmployee.onclick = function() { startCaptureProcess('employee'); };
    } else if (valid) {
        btnRegisterEmployee.disabled = true;
        btnRegisterEmployee.textContent = 'START CAPTURE';
        btnRegisterEmployee.onclick = null;
    } else {
        btnRegisterEmployee.disabled = true;
        btnRegisterEmployee.onclick = null;
    }
}

if (tabEmployee) tabEmployee.addEventListener('click', function() { switchTab('employee'); });
if (tabManforce) tabManforce.addEventListener('click', function() { switchTab('manforce'); });

// Swipe on tab bar to switch
var tabBar = document.querySelector('.register-tabs');
if (tabBar) {
    var touchStartX = 0;
    tabBar.addEventListener('touchstart', function(e) { touchStartX = e.touches[0].clientX; }, { passive: true });
    tabBar.addEventListener('touchend', function(e) {
        var dx = (e.changedTouches[0].clientX - touchStartX);
        if (Math.abs(dx) > 50) {
            if (dx > 0 && activeTab === 'manforce') switchTab('employee');
            else if (dx < 0 && activeTab === 'employee') switchTab('manforce');
        }
    }, { passive: true });
}

[manforceAadhaar, manforceName, manforceMobile].forEach(function(input) {
    if (input) input.addEventListener('input', function() { updateManforceButton(); });
});
if (employeeCode) employeeCode.addEventListener('input', function() { updateEmployeeButton(); });

function updateProgress(percent) {
    if (progressCircle) {
        progressCircle.style.strokeDashoffset = 113 - (113 * percent);
    }
}

function captureFrameAsDataUrl() {
    if (!video || video.readyState < 2) return null;
    var canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    return canvas.toDataURL('image/jpeg', 0.8);
}

function runFaceCheck() {
    if (isCapturing || !video || video.readyState < 2) return;
    var frame = captureFrameAsDataUrl();
    if (!frame) return;
    fetch('/api/face-check', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame: frame })
    })
    .then(function(r) { return r.json(); })
    .then(function(data) {
        var wasReady = faceReady;
        faceReady = data.ready === true;
        if (faceReady && !wasReady) updateVisibleButton();
        if (!faceReady && wasReady) updateVisibleButton();
        var statusEl = activeTab === 'manforce' ? statusTextManforce : statusTextEmployee;
        var globalStatus = statusText;
        if (getFramesFor(activeTab).length === REQUIRED_FRAMES) return;
        if (data.ready) {
            if (statusEl) statusEl.textContent = 'Face detected — click START CAPTURE';
            if (globalStatus) globalStatus.textContent = 'Face detected — click START CAPTURE';
        } else if (data.face_detected) {
            if (statusEl) statusEl.textContent = 'Move closer and center your face';
            if (globalStatus) globalStatus.textContent = 'Move closer and center your face';
        } else {
            if (statusEl) statusEl.textContent = 'Position your face in the frame';
            if (globalStatus) globalStatus.textContent = 'Position your face in the frame';
        }
    })
    .catch(function() {});
}

function startFaceCheckPolling() {
    stopFaceCheckPolling();
    faceCheckInterval = setInterval(runFaceCheck, FACE_CHECK_POLL_MS);
    runFaceCheck();
}

function stopFaceCheckPolling() {
    if (faceCheckInterval) {
        clearInterval(faceCheckInterval);
        faceCheckInterval = null;
    }
}

function startCaptureProcess(target) {
    if (isCapturing) return;
    captureTarget = target;
    isCapturing = true;
    stopFaceCheckPolling();
    setFramesFor(target, []);
    updateVisibleButton();
    var statusEl = target === 'manforce' ? statusTextManforce : statusTextEmployee;
    if (statusEl) statusEl.textContent = 'Look at the camera and blink naturally...';
    if (statusText) statusText.textContent = 'Look at the camera and blink naturally during capture.';

    var count = 0;
    var slotRetries = 0;
    function trySlot() {
        if (count >= REQUIRED_FRAMES) {
            finishCapture(target);
            return;
        }
        var frame = captureFrameAsDataUrl();
        if (!frame) {
            setTimeout(trySlot, CAPTURE_INTERVAL_MS);
            return;
        }
        fetch('/api/face-check', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frame: frame })
        })
        .then(function(r) { return r.json(); })
        .then(function(data) {
            if (data.face_detected) {
                var frames = getFramesFor(target);
                frames.push(frame);
                setFramesFor(target, frames);
                count++;
                slotRetries = 0;
                updateProgress(count / REQUIRED_FRAMES);
                var pct = Math.round((count / REQUIRED_FRAMES) * 100);
                if (statusEl) statusEl.textContent = 'Scanning... ' + pct + '%';
                if (statusText) statusText.textContent = 'Scanning... ' + pct + '%';
                setTimeout(trySlot, CAPTURE_INTERVAL_MS);
            } else if (slotRetries < MAX_RETRIES_PER_SLOT) {
                slotRetries++;
                if (statusEl) statusEl.textContent = 'Face not in frame — hold still...';
                if (statusText) statusText.textContent = 'Face not in frame — hold still...';
                setTimeout(trySlot, CAPTURE_INTERVAL_MS);
            } else {
                var frames = getFramesFor(target);
                frames.push(frame);
                setFramesFor(target, frames);
                count++;
                slotRetries = 0;
                updateProgress(count / REQUIRED_FRAMES);
                var pct = Math.round((count / REQUIRED_FRAMES) * 100);
                if (statusEl) statusEl.textContent = 'Scanning... ' + pct + '%';
                if (statusText) statusText.textContent = 'Scanning... ' + pct + '%';
                setTimeout(trySlot, CAPTURE_INTERVAL_MS);
            }
        })
        .catch(function() {
            setTimeout(trySlot, CAPTURE_INTERVAL_MS);
        });
    }
    setTimeout(trySlot, CAPTURE_INTERVAL_MS);
}

function finishCapture(target) {
    isCapturing = false;
    captureTarget = null;
    startFaceCheckPolling();
    var statusEl = target === 'manforce' ? statusTextManforce : statusTextEmployee;
    if (statusEl) {
        statusEl.textContent = 'Face captured ✓';
        statusEl.style.color = 'var(--color-primary)';
    }
    if (statusText) {
        statusText.textContent = 'Face captured ✓';
        statusText.style.color = 'var(--color-primary)';
    }
    updateVisibleButton();
}

async function submitRegistration(userType) {
    var isManforce = userType === 'manforce';
    var btn = isManforce ? btnRegisterManforce : btnRegisterEmployee;
    var frames = getFramesFor(userType);
    if (!btn || !frames || frames.length === 0) return;

    btn.disabled = true;
    btn.textContent = 'REGISTERING...';

    var payload;
    if (isManforce) {
        if (!manforceAadhaar || !manforceName || !manforceMobile) return;
        payload = {
            user_type: 'manforce',
            aadhaar: manforceAadhaar.value.trim(),
            name: manforceName.value.trim(),
            mobile: manforceMobile.value.trim(),
            frames: frames
        };
    } else {
        if (!employeeCode) return;
        payload = {
            user_type: 'employee',
            employee_id: employeeCode.value.trim(),
            frames: frames
        };
    }

    try {
        var res = await fetch('/api/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        var data = await res.json();

        if (data.status === 'registered') {
            document.getElementById('registerForm').style.display = 'none';
            document.getElementById('successState').style.display = 'block';
            var label = isManforce ? payload.name + ' — ' + payload.aadhaar : payload.employee_id;
            document.getElementById('successChip').textContent = label;
        } else if (data.status === 'spoof') {
            showSpoofToast(data);
            btn.disabled = false;
            btn.textContent = 'Register';
        } else {
            var msg = data.message || 'Unknown error';
            if (msg.indexOf('already registered') !== -1 || msg.indexOf('already registered to') !== -1) {
                if (typeof showSnackbar === 'function') showSnackbar('Duplicate registration — this face is already registered.', 'error');
            } else if (msg.indexOf('No face detected') !== -1 || msg.indexOf('no face') !== -1) {
                if (typeof showSnackbar === 'function') showSnackbar('Clear photo — ensure your face is visible and well lit.', 'info');
            } else {
                if (typeof showSnackbar === 'function') showSnackbar(msg, 'error');
            }
            btn.disabled = false;
            btn.textContent = 'Register';
        }
    } catch (e) {
        if (typeof showSnackbar === 'function') showSnackbar('Network error. Please try again.', 'error');
        btn.disabled = false;
        btn.textContent = 'Register';
    }
}

function showSpoofToast(data) {
    var msg = data.reason || data.message || 'Use a live face, not a photo or screen.';
    if (spoofToastMessage) spoofToastMessage.textContent = msg;
    if (spoofToast) spoofToast.classList.add('show');
    if (toastDismissTimer) clearTimeout(toastDismissTimer);
    toastDismissTimer = setTimeout(function() {
        if (spoofToast) spoofToast.classList.remove('show');
        toastDismissTimer = null;
    }, 4500);
}

// Init: ensure Employee tab/panel active and start live face feedback
switchTab('employee');
startFaceCheckPolling();
