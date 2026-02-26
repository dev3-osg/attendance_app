const video = document.getElementById('videoFeed');

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'user', width: 640, height: 480 } 
        });
        video.srcObject = stream;
    } catch (err) {
        console.error("Camera error:", err);
    }
}
