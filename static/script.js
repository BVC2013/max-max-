const video = document.getElementById('video');
const predictionText = document.getElementById('prediction');

navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
    video.srcObject = stream;
});

function captureAndSendFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, 224, 224);
    const dataUrl = canvas.toDataURL('image/jpeg');

    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl })
    })
    .then(res => res.json())
    .then(data => {
        predictionText.textContent = data.class;
    })
    .catch(err => {
        predictionText.textContent = 'Error';
        console.error(err);
    });
}

setInterval(captureAndSendFrame, 1000);
