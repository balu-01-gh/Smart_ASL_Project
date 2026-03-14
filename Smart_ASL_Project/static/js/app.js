/**
 * Smart ASL Recognition - Frontend Application
 * Client-side MediaPipe Hands for INSTANT landmark detection,
 * server-side LSTM for ASL word prediction.
 */

// =====================
// State
// =====================
let stream = null;
let isRunning = false;
let currentPrediction = null;
let sentence = [];
let missedFrames = 0;
let autoAddEnabled = false;
let stablePredictionCount = 0;
let lastStablePrediction = null;
let predictionHistory = [];
let fpsFrameCount = 0;
let fpsLastTime = performance.now();
let currentFps = 0;
let clientHandDetected = false;
let mpHands = null;
let mpCamera = null;
let isSendingToServer = false;
let lastLandmarks = [];

const CAPTURE_DELAY_MS = 100;   // 100ms → fills 13-frame buffer in ~1.3s instead of 6.5s
const AUTO_ADD_THRESHOLD = 4;
const MAX_HISTORY = 50;

// =====================
// DOM Elements
// =====================
const webcam = document.getElementById("webcam");
const captureCanvas = document.getElementById("captureCanvas");
const overlayCanvas = document.getElementById("overlayCanvas");
const cameraOverlay = document.getElementById("cameraOverlay");
const handIndicator = document.getElementById("handIndicator");
const frameProgress = document.getElementById("frameProgress");
const progressFill = document.getElementById("progressFill");
const frameCount = document.getElementById("frameCount");
const predictionWord = document.getElementById("predictionWord");
const confidenceFill = document.getElementById("confidenceFill");
const confidenceValue = document.getElementById("confidenceValue");
const top3List = document.getElementById("top3List");
const sentenceDisplay = document.getElementById("sentenceDisplay");
const statusIndicator = document.getElementById("statusIndicator");
const statusText = document.getElementById("statusText");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const fpsCounter = document.getElementById("fpsCounter");
const autoAddIndicator = document.getElementById("autoAddIndicator");
const autoAddToggle = document.getElementById("autoAddToggle");
const historyTimeline = document.getElementById("historyTimeline");
const toastContainer = document.getElementById("toastContainer");


// =====================
// Client-Side MediaPipe Hands Setup
// =====================
let mpReady = false;
let mpInitializing = false;

async function initMediaPipeHands() {
    if (mpInitializing) return;
    mpInitializing = true;

    console.log("[MP] Initializing client-side MediaPipe Hands...");
    setStatus("detecting", "Loading hand detection model (first time may take 10-20s)...");
    showToast("Downloading hand detection model...", "info");

    try {
        mpHands = new Hands({
            locateFile: (file) => {
                console.log("[MP] Loading file:", file);
                return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`;
            }
        });

        mpHands.setOptions({
            maxNumHands: 2,
            modelComplexity: 1,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        mpHands.onResults(onHandResults);

        // Warm up: send a blank frame to trigger WASM download
        const warmupCanvas = document.createElement("canvas");
        warmupCanvas.width = 64;
        warmupCanvas.height = 64;
        warmupCanvas.getContext("2d").fillRect(0, 0, 64, 64);
        await mpHands.send({ image: warmupCanvas });

        mpReady = true;
        console.log("[MP] MediaPipe Hands ready!");
        setStatus("online", "Hand detection ready — show your hand");
        showToast("Hand detection model loaded!", "success");
    } catch (err) {
        console.error("[MP] Init error:", err);
        setStatus("offline", "Hand detection failed to load");
        showToast("MediaPipe failed: " + err.message, "error");
        mpInitializing = false;
    }
}

/**
 * Called every frame by MediaPipe with detection results.
 * This runs at full camera FPS (~30) for instant feedback.
 */
function onHandResults(results) {
    if (!isRunning) return;

    // Update FPS counter
    fpsFrameCount++;
    const now = performance.now();
    const elapsed = now - fpsLastTime;
    if (elapsed >= 1000) {
        currentFps = Math.round((fpsFrameCount / elapsed) * 1000);
        fpsFrameCount = 0;
        fpsLastTime = now;
        fpsCounter.textContent = `${currentFps} FPS`;
        fpsCounter.className = `fps-counter ${currentFps >= 15 ? "fps-good" : currentFps >= 8 ? "fps-ok" : "fps-slow"}`;
    }

    const hasHands = results.multiHandLandmarks && results.multiHandLandmarks.length > 0;

    if (hasHands) {
        clientHandDetected = true;
        missedFrames = 0;
        handIndicator.classList.remove("hidden");
        frameProgress.classList.remove("hidden");

        // Convert landmarks to our format for drawing
        lastLandmarks = results.multiHandLandmarks.map(hand =>
            hand.map(lm => ({ x: lm.x, y: lm.y }))
        );

        // Draw landmarks IMMEDIATELY (no server round-trip!)
        drawHandLandmarks(lastLandmarks);

        if (!isSendingToServer) {
            setStatus("detecting", "Hand detected — analyzing...");
        }
    } else {
        missedFrames++;
        if (missedFrames >= 8) {
            clientHandDetected = false;
            handIndicator.classList.add("hidden");
            frameProgress.classList.add("hidden");
            lastLandmarks = [];
            drawHandLandmarks([]);  // Clear overlay
            if (!isSendingToServer) {
                setStatus("online", "No hand detected — show your hand");
            }
        }
    }
}


// =====================
// Camera Controls
// =====================
async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user"
            }
        });

        webcam.srcObject = stream;
        await webcam.play();

        cameraOverlay.classList.add("hidden");
        startBtn.disabled = true;
        stopBtn.disabled = false;
        isRunning = true;
        missedFrames = 0;
        fpsFrameCount = 0;
        fpsLastTime = performance.now();

        setStatus("online", "Camera active — initializing hand detection...");
        showToast("Camera started", "success");

        // Initialize MediaPipe if not done (await WASM load)
        if (!mpReady) {
            await initMediaPipeHands();
        }

        // Only start loops if MediaPipe loaded successfully
        if (mpReady) {
            startClientDetectionLoop();
            startServerPredictionLoop();
        } else {
            showToast("Hand detection not available — using server-only mode", "info");
            // Fallback: server-only mode
            startServerOnlyLoop();
        }

    } catch (err) {
        console.error("Camera error:", err);
        setStatus("offline", "Camera access denied");
        showToast("Camera access denied. Check permissions.", "error");
    }
}

function stopCamera() {
    isRunning = false;

    // Stop Camera utility if running
    if (mpCamera) {
        mpCamera.stop();
        mpCamera = null;
    }

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    webcam.srcObject = null;
    cameraOverlay.classList.remove("hidden");
    handIndicator.classList.add("hidden");
    frameProgress.classList.add("hidden");
    autoAddIndicator.classList.add("hidden");

    startBtn.disabled = false;
    stopBtn.disabled = true;

    setStatus("offline", "Camera off");
    fpsCounter.textContent = "";
    clientHandDetected = false;
    lastLandmarks = [];
    drawHandLandmarks([]);
    resetPredictionDisplay();
    showToast("Camera stopped", "info");
}

function toggleCamera() {
    if (isRunning) {
        stopCamera();
    } else {
        startCamera();
    }
}


// =====================
// Client-Side Detection Loop (using Camera utility for proper pacing)
// =====================
function startClientDetectionLoop() {
    console.log("[MP] Starting client-side detection loop via Camera utility");

    if (mpCamera) {
        mpCamera.stop();
    }

    mpCamera = new Camera(webcam, {
        onFrame: async () => {
            if (!isRunning || !mpReady) return;
            try {
                await mpHands.send({ image: webcam });
            } catch (err) {
                console.warn("[MP] Frame error:", err);
            }
        },
        width: 640,
        height: 480
    });

    mpCamera.start()
        .then(() => console.log("[MP] Camera utility running"))
        .catch(err => console.warn("[MP] Camera utility error:", err));
}

// Fallback: server-only mode if client-side MediaPipe fails
async function startServerOnlyLoop() {
    console.log("[Server] Fallback: server-only detection mode");
    while (isRunning) {
        await sendFrameToServer();
        await new Promise(r => setTimeout(r, CAPTURE_DELAY_MS));
    }
}


// =====================
// Server Prediction Loop (ASL recognition)
// =====================
async function startServerPredictionLoop() {
    console.log("[Server] Starting prediction loop");

    while (isRunning) {
        if (clientHandDetected) {
            await sendFrameToServer();
        }
        await new Promise(r => setTimeout(r, CAPTURE_DELAY_MS));
    }
}

async function sendFrameToServer() {
    if (!isRunning || !webcam.videoWidth) return;

    isSendingToServer = true;

    captureCanvas.width = webcam.videoWidth;
    captureCanvas.height = webcam.videoHeight;
    const ctx = captureCanvas.getContext("2d");
    ctx.drawImage(webcam, 0, 0);

    const frameData = captureCanvas.toDataURL("image/jpeg", 0.90);

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ frame: frameData })
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();
        handleServerPrediction(data);

    } catch (err) {
        console.error("Prediction error:", err);
        setStatus("detecting", "Connection issue — retrying...");
    }

    isSendingToServer = false;
}


// =====================
// Handle Server Prediction Response
// =====================
function handleServerPrediction(data) {
    if (data.error) {
        setStatus("detecting", `Error: ${data.error}`);
        return;
    }

    // Frame progress
    const bufferSize = data.buffer_size || 0;
    const required = data.required_frames || 13;
    const pct = Math.round((bufferSize / required) * 100);
    progressFill.style.width = pct + "%";
    frameCount.textContent = `${bufferSize}/${required} frames`;

    // Server-side processing time
    if (data.process_ms) {
        const existing = fpsCounter.textContent;
        const fpsOnly = existing.split("·")[0].trim();
        fpsCounter.textContent = `${fpsOnly} · ${data.process_ms}ms`;
    }

    // Prediction
    if (data.prediction) {
        const isNewPrediction = currentPrediction !== data.prediction;
        currentPrediction = data.prediction;
        displayPrediction(data.prediction, data.confidence);
        setStatus("online", `Recognized: ${data.prediction}`);

        if (data.top3) {
            displayTop3(data.top3);
        }

        if (isNewPrediction) {
            addToHistory(data.prediction, data.confidence);
            showToast(`Recognized: ${data.prediction}`, "success");
        }

        handleAutoAdd(data.prediction, data.confidence);

    } else if (data.hand_detected) {
        setStatus("detecting", data.message || "Collecting frames...");
        stablePredictionCount = 0;
    } else {
        // Server didn't detect hand (might differ from client slightly)
        stablePredictionCount = 0;
    }
}


// =====================
// Auto-Add to Sentence
// =====================
function handleAutoAdd(word, confidence) {
    if (!autoAddEnabled) {
        autoAddIndicator.classList.add("hidden");
        return;
    }

    if (word === lastStablePrediction) {
        stablePredictionCount++;
    } else {
        stablePredictionCount = 1;
        lastStablePrediction = word;
    }

    if (stablePredictionCount > 1) {
        autoAddIndicator.classList.remove("hidden");
        autoAddIndicator.innerHTML = `<i class="fas fa-magic"></i> Auto-add: ${stablePredictionCount}/${AUTO_ADD_THRESHOLD}`;
    }

    if (stablePredictionCount >= AUTO_ADD_THRESHOLD) {
        if (sentence.length === 0 || sentence[sentence.length - 1] !== word) {
            sentence.push(word);
            renderSentence();
            showToast(`Auto-added: ${word}`, "info");
        }
        stablePredictionCount = 0;
        lastStablePrediction = null;
        autoAddIndicator.classList.add("hidden");
    }
}

function toggleAutoAdd() {
    autoAddEnabled = autoAddToggle.checked;
    stablePredictionCount = 0;
    lastStablePrediction = null;
    if (!autoAddEnabled) {
        autoAddIndicator.classList.add("hidden");
    }
    showToast(autoAddEnabled ? "Auto-add enabled" : "Auto-add disabled", "info");
}


// =====================
// Prediction History
// =====================
function addToHistory(word, confidence) {
    const entry = {
        word,
        confidence: Math.round(confidence * 100),
        time: new Date().toLocaleTimeString()
    };
    predictionHistory.unshift(entry);
    if (predictionHistory.length > MAX_HISTORY) {
        predictionHistory.pop();
    }
    renderHistory();
}

function renderHistory() {
    if (predictionHistory.length === 0) {
        historyTimeline.innerHTML = `
            <div class="history-empty">
                <i class="fas fa-clock"></i>
                <p>Predictions will appear here as you sign</p>
            </div>
        `;
        return;
    }

    historyTimeline.innerHTML = predictionHistory
        .map((entry, idx) => `
            <div class="history-item ${idx === 0 ? 'latest' : ''}" style="animation-delay: ${idx * 0.03}s">
                <span class="history-word">${entry.word}</span>
                <span class="history-conf ${entry.confidence >= 70 ? 'high' : entry.confidence >= 40 ? 'mid' : 'low'}">${entry.confidence}%</span>
                <span class="history-time">${entry.time}</span>
            </div>
        `)
        .join("");
}

function clearHistory() {
    predictionHistory = [];
    renderHistory();
    showToast("History cleared", "info");
}


// =====================
// Display Functions
// =====================
function displayPrediction(word, confidence) {
    predictionWord.innerHTML = word;
    predictionWord.classList.add("prediction-pop");
    setTimeout(() => predictionWord.classList.remove("prediction-pop"), 300);

    const confPct = Math.round(confidence * 100);
    confidenceFill.style.width = confPct + "%";
    confidenceValue.textContent = confPct + "%";

    if (confPct >= 70) {
        confidenceFill.style.background = "linear-gradient(135deg, #00D9A6, #6C63FF)";
    } else if (confPct >= 40) {
        confidenceFill.style.background = "linear-gradient(135deg, #FFAA00, #FF6B6B)";
    } else {
        confidenceFill.style.background = "linear-gradient(135deg, #FF6B6B, #FF4444)";
    }
}

function displayTop3(top3) {
    let html = "";
    top3.forEach((item, index) => {
        const confPct = Math.round(item.confidence * 100);
        const barWidth = Math.max(confPct, 5);
        html += `
            <div class="top3-item">
                <span class="rank">#${index + 1}</span>
                <span class="word">${item.word}</span>
                <div class="top3-bar-track">
                    <div class="top3-bar-fill" style="width: ${barWidth}%"></div>
                </div>
                <span class="conf">${confPct}%</span>
            </div>
        `;
    });
    top3List.innerHTML = html;
}

function resetPredictionDisplay() {
    predictionWord.innerHTML = '<span class="waiting">Waiting...</span>';
    confidenceFill.style.width = "0%";
    confidenceValue.textContent = "0%";
    top3List.innerHTML = `
        <div class="top3-item empty"><span class="rank">#1</span><span class="word">—</span><span class="conf">—</span></div>
        <div class="top3-item empty"><span class="rank">#2</span><span class="word">—</span><span class="conf">—</span></div>
        <div class="top3-item empty"><span class="rank">#3</span><span class="word">—</span><span class="conf">—</span></div>
    `;
}

function setStatus(state, text) {
    statusIndicator.className = `status-dot ${state}`;
    statusText.textContent = text;
}


// =====================
// Toast Notifications
// =====================
function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;

    const icons = { success: "check-circle", error: "exclamation-circle", info: "info-circle" };
    toast.innerHTML = `<i class="fas fa-${icons[type] || icons.info}"></i> ${message}`;

    toastContainer.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add("toast-visible"));

    setTimeout(() => {
        toast.classList.remove("toast-visible");
        toast.classList.add("toast-exit");
        setTimeout(() => toast.remove(), 300);
    }, 2500);
}


// =====================
// Reset Prediction (API call)
// =====================
async function resetPrediction() {
    try {
        await fetch("/api/reset", { method: "POST" });
    } catch (err) {
        console.error("Reset error:", err);
    }
    currentPrediction = null;
    stablePredictionCount = 0;
    lastStablePrediction = null;
    resetPredictionDisplay();
    handIndicator.classList.add("hidden");
    frameProgress.classList.add("hidden");
    autoAddIndicator.classList.add("hidden");
    progressFill.style.width = "0%";
    showToast("Buffers reset", "info");
}


// =====================
// Sentence Builder
// =====================
function addWordToSentence() {
    if (!currentPrediction) {
        showToast("No prediction to add", "error");
        return;
    }
    sentence.push(currentPrediction);
    renderSentence();
    showToast(`Added: ${currentPrediction}`, "success");
}

function undoLastWord() {
    if (sentence.length === 0) return;
    const removed = sentence.pop();
    renderSentence();
    showToast(`Removed: ${removed}`, "info");
}

function clearSentence() {
    if (sentence.length === 0) return;
    sentence = [];
    renderSentence();
    showToast("Sentence cleared", "info");
}

function renderSentence() {
    if (sentence.length === 0) {
        sentenceDisplay.innerHTML = '<span class="placeholder">Signs will appear here...</span>';
        return;
    }
    sentenceDisplay.innerHTML = sentence
        .map((word, idx) => `<span class="sentence-word" onclick="removeWordAt(${idx})" title="Click to remove">${word}</span>`)
        .join(" ");
}

function removeWordAt(index) {
    const removed = sentence.splice(index, 1)[0];
    renderSentence();
    showToast(`Removed: ${removed}`, "info");
}

function copySentence() {
    if (sentence.length === 0) {
        showToast("Nothing to copy", "error");
        return;
    }
    const text = sentence.join(" ");
    navigator.clipboard.writeText(text).then(() => {
        showToast("Sentence copied!", "success");
    }).catch(() => {
        showToast("Copy failed", "error");
    });
}

function speakSentence() {
    if (sentence.length === 0) {
        showToast("Nothing to speak", "error");
        return;
    }
    const text = sentence.join(" ");
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    utterance.pitch = 1;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
    showToast("Speaking...", "info");
}


// =====================
// Keyboard Shortcuts
// =====================
document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

    if (e.code === "Space" && !e.ctrlKey && !e.shiftKey) {
        e.preventDefault();
        toggleCamera();
    }
    if (e.code === "KeyR" && !e.ctrlKey && !e.shiftKey) {
        e.preventDefault();
        resetPrediction();
    }
    if (e.code === "Enter" && !e.ctrlKey && !e.shiftKey) {
        e.preventDefault();
        addWordToSentence();
    }
    if (e.code === "KeyZ" && e.ctrlKey && !e.shiftKey) {
        e.preventDefault();
        undoLastWord();
    }
    if (e.code === "KeyC" && e.ctrlKey && e.shiftKey) {
        e.preventDefault();
        copySentence();
    }
    if (e.code === "KeyS" && !e.ctrlKey && !e.shiftKey) {
        e.preventDefault();
        speakSentence();
    }
});


// =====================
// Word Grid (Supported Words)
// =====================
async function loadWords() {
    try {
        const response = await fetch("/api/labels");
        const data = await response.json();
        const grid = document.getElementById("wordGrid");
        const countBadge = document.getElementById("wordCount");
        if (data.labels && data.labels.length > 0) {
            countBadge.textContent = data.count;
            grid.innerHTML = data.labels
                .map(word => `<div class="word-chip" data-word="${word.toLowerCase()}">${word}</div>`)
                .join("");
        }
    } catch (err) {
        console.error("Failed to load words:", err);
        document.getElementById("wordGrid").innerHTML =
            '<p style="color: var(--text-muted); grid-column: 1/-1;">Start the server to see supported words.</p>';
    }
}

function filterWords() {
    const query = document.getElementById("wordSearch").value.toLowerCase().trim();
    document.querySelectorAll(".word-chip").forEach(chip => {
        const word = chip.getAttribute("data-word");
        chip.classList.toggle("hidden", query && !word.includes(query));
    });
}


// =====================
// Hand Landmark Drawing (used by client-side MediaPipe)
// =====================
const HAND_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20],
    [5,9],[9,13],[13,17]
];

const FINGERTIPS = [4, 8, 12, 16, 20];

function drawHandLandmarks(landmarksList) {
    if (!webcam.videoWidth) return;

    overlayCanvas.width = webcam.videoWidth;
    overlayCanvas.height = webcam.videoHeight;
    const ctx = overlayCanvas.getContext("2d");
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    if (!landmarksList || landmarksList.length === 0) return;

    const w = overlayCanvas.width;
    const h = overlayCanvas.height;

    landmarksList.forEach(landmarks => {
        // Connections with gradient
        HAND_CONNECTIONS.forEach(([i, j]) => {
            const a = landmarks[i];
            const b = landmarks[j];
            if (a && b) {
                const gradient = ctx.createLinearGradient(a.x * w, a.y * h, b.x * w, b.y * h);
                gradient.addColorStop(0, "rgba(0, 217, 166, 0.9)");
                gradient.addColorStop(1, "rgba(108, 99, 255, 0.9)");
                ctx.strokeStyle = gradient;
                ctx.lineWidth = 2.5;
                ctx.lineCap = "round";
                ctx.beginPath();
                ctx.moveTo(a.x * w, a.y * h);
                ctx.lineTo(b.x * w, b.y * h);
                ctx.stroke();
            }
        });

        // Points with fingertip glow
        landmarks.forEach((lm, idx) => {
            const isTip = FINGERTIPS.includes(idx);
            const radius = isTip ? 6 : 3.5;

            if (isTip) {
                ctx.beginPath();
                ctx.arc(lm.x * w, lm.y * h, 10, 0, 2 * Math.PI);
                ctx.fillStyle = "rgba(108, 99, 255, 0.25)";
                ctx.fill();
            }

            ctx.beginPath();
            ctx.arc(lm.x * w, lm.y * h, radius, 0, 2 * Math.PI);
            ctx.fillStyle = isTip ? "#6C63FF" : "#00D9A6";
            ctx.fill();
            ctx.strokeStyle = "#fff";
            ctx.lineWidth = 1.5;
            ctx.stroke();
        });
    });
}


// =====================
// Initialize
// =====================
document.addEventListener("DOMContentLoaded", () => {
    loadWords();

    const savedAutoAdd = localStorage.getItem("autoAddEnabled");
    if (savedAutoAdd === "true") {
        autoAddEnabled = true;
        autoAddToggle.checked = true;
    }

    autoAddToggle.addEventListener("change", () => {
        localStorage.setItem("autoAddEnabled", autoAddToggle.checked);
    });
});
