/* ========================================================
   FSR Matrix — Frontend Logic
   Heatmap renderer, dashboard, calibration panel
   ======================================================== */

(function () {
    "use strict";

    /* --- DOM References --- */
    const DOM = {
        canvas: document.getElementById("heatmap-canvas"),
        weightValue: document.getElementById("weight-value"),
        timerValue: document.getElementById("timer-value"),
        timerBar: document.getElementById("timer-bar"),
        timerCard: document.getElementById("timer-card"),
        statusDot: document.getElementById("status-dot"),
        statusText: document.getElementById("status-text"),
        gridSize: document.getElementById("grid-size"),
        fpsValue: document.getElementById("fps-value"),
        alertOverlay: document.getElementById("alert-overlay"),
        alertTime: document.getElementById("alert-time"),
        heatmapContainer: document.getElementById("heatmap-container"),
    };

    const ctx = DOM.canvas.getContext("2d");

    /* --- State --- */
    let lastFrameTime = performance.now();
    let frameCount = 0;
    let currentFps = 0;
    let alertDismissed = false;
    let lastAlertState = false;
    let debounceTimers = {};

    /* --- Color Palette (Spectral: blue → cyan → green → yellow → red) --- */
    const HEATMAP_COLORS = [
        [15, 15, 60],      /* 0.0 — deep navy */
        [20, 50, 140],     /* 0.15 — dark blue */
        [10, 120, 200],    /* 0.25 — blue */
        [10, 190, 200],    /* 0.35 — cyan */
        [30, 200, 120],    /* 0.45 — teal-green */
        [100, 220, 60],    /* 0.55 — green */
        [200, 230, 40],    /* 0.65 — yellow-green */
        [250, 210, 30],    /* 0.75 — yellow */
        [255, 140, 20],    /* 0.85 — orange */
        [240, 50, 30],     /* 0.95 — red */
        [180, 10, 50],     /* 1.0  — dark red */
    ];

    function interpolateColor(value) {
        const t = Math.max(0, Math.min(1, value));
        const idx = t * (HEATMAP_COLORS.length - 1);
        const lo = Math.floor(idx);
        const hi = Math.min(lo + 1, HEATMAP_COLORS.length - 1);
        const frac = idx - lo;

        const r = Math.round(HEATMAP_COLORS[lo][0] + frac * (HEATMAP_COLORS[hi][0] - HEATMAP_COLORS[lo][0]));
        const g = Math.round(HEATMAP_COLORS[lo][1] + frac * (HEATMAP_COLORS[hi][1] - HEATMAP_COLORS[lo][1]));
        const b = Math.round(HEATMAP_COLORS[lo][2] + frac * (HEATMAP_COLORS[hi][2] - HEATMAP_COLORS[lo][2]));
        return [r, g, b];
    }

    /* --- Heatmap Rendering --- */
    function renderHeatmap(heatmapData, rows, cols) {
        if (!heatmapData || rows === 0 || cols === 0) return;

        const container = DOM.heatmapContainer;
        const maxW = container.clientWidth - 40;
        const maxH = container.clientHeight - 60;
        const aspect = cols / rows;

        let canvasW, canvasH;
        if (maxW / maxH > aspect) {
            canvasH = maxH;
            canvasW = Math.round(canvasH * aspect);
        } else {
            canvasW = maxW;
            canvasH = Math.round(canvasW / aspect);
        }

        canvasW = Math.max(canvasW, 100);
        canvasH = Math.max(canvasH, 100);

        DOM.canvas.width = canvasW;
        DOM.canvas.height = canvasH;

        const cellW = canvasW / cols;
        const cellH = canvasH / rows;

        /* Render with smooth interpolation using upscaled off-screen canvas */
        const offscreen = document.createElement("canvas");
        offscreen.width = cols;
        offscreen.height = rows;
        const offCtx = offscreen.getContext("2d");
        const imgData = offCtx.createImageData(cols, rows);

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const val = heatmapData[r] ? (heatmapData[r][c] || 0) : 0;
                const [cr, cg, cb] = interpolateColor(val);
                const idx = (r * cols + c) * 4;
                imgData.data[idx] = cr;
                imgData.data[idx + 1] = cg;
                imgData.data[idx + 2] = cb;
                imgData.data[idx + 3] = 255;
            }
        }
        offCtx.putImageData(imgData, 0, 0);

        /* Smooth upscale */
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = "high";
        ctx.clearRect(0, 0, canvasW, canvasH);

        /* Draw rounded clip */
        const radius = 12;
        ctx.beginPath();
        ctx.moveTo(radius, 0);
        ctx.lineTo(canvasW - radius, 0);
        ctx.quadraticCurveTo(canvasW, 0, canvasW, radius);
        ctx.lineTo(canvasW, canvasH - radius);
        ctx.quadraticCurveTo(canvasW, canvasH, canvasW - radius, canvasH);
        ctx.lineTo(radius, canvasH);
        ctx.quadraticCurveTo(0, canvasH, 0, canvasH - radius);
        ctx.lineTo(0, radius);
        ctx.quadraticCurveTo(0, 0, radius, 0);
        ctx.closePath();
        ctx.clip();

        ctx.drawImage(offscreen, 0, 0, canvasW, canvasH);

        /* Grid lines */
        ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
        ctx.lineWidth = 1;
        for (let r = 1; r < rows; r++) {
            const y = Math.round(r * cellH) + 0.5;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(canvasW, y);
            ctx.stroke();
        }
        for (let c = 1; c < cols; c++) {
            const x = Math.round(c * cellW) + 0.5;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, canvasH);
            ctx.stroke();
        }

        /* Cell value labels */
        ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
        const fontSize = Math.max(10, Math.min(cellW * 0.28, 18));
        ctx.font = `500 ${fontSize}px 'JetBrains Mono', monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const val = heatmapData[r] ? (heatmapData[r][c] || 0) : 0;
                if (val > 0.01) {
                    const x = c * cellW + cellW / 2;
                    const y = r * cellH + cellH / 2;
                    const pct = Math.round(val * 100);
                    ctx.fillStyle = val > 0.5
                        ? "rgba(0, 0, 0, 0.6)"
                        : "rgba(255, 255, 255, 0.75)";
                    ctx.fillText(`${pct}%`, x, y);
                }
            }
        }
    }

    /* --- Dashboard Updates --- */
    function updateWeight(kg) {
        DOM.weightValue.textContent = kg.toFixed(2);
    }

    function formatTimer(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, "0")}`;
    }

    function updateTimer(seconds, timeoutSeconds, isAlert) {
        DOM.timerValue.textContent = formatTimer(seconds);
        const pct = Math.min((seconds / timeoutSeconds) * 100, 100);
        DOM.timerBar.style.width = `${pct}%`;

        if (isAlert) {
            DOM.timerCard.classList.add("alert-active");
        } else {
            DOM.timerCard.classList.remove("alert-active");
        }
    }

    function updateStatus(status) {
        const connected = status === "conectado";
        DOM.statusDot.className = connected
            ? "status-dot connected"
            : "status-dot disconnected";
        DOM.statusText.textContent = status;
    }

    function updateFps() {
        frameCount++;
        const now = performance.now();
        const elapsed = now - lastFrameTime;
        if (elapsed >= 1000) {
            currentFps = Math.round((frameCount * 1000) / elapsed);
            DOM.fpsValue.textContent = currentFps;
            frameCount = 0;
            lastFrameTime = now;
        }
    }

    /* --- Alert --- */
    function showAlert(seconds) {
        if (alertDismissed) return;
        DOM.alertTime.textContent = formatTimer(seconds);
        DOM.alertOverlay.classList.remove("hidden");
    }

    function hideAlert() {
        DOM.alertOverlay.classList.add("hidden");
    }

    window.dismissAlert = function () {
        alertDismissed = true;
        hideAlert();
    };

    /* --- Calibration Panel --- */
    function initCalibrationPanel() {
        const inputs = document.querySelectorAll(".calib-field input");
        inputs.forEach(function (input) {
            input.addEventListener("input", function () {
                const key = input.dataset.key;
                const value = parseFloat(input.value);
                if (isNaN(value)) return;

                clearTimeout(debounceTimers[key]);
                debounceTimers[key] = setTimeout(function () {
                    eel.update_calibration(key, value)(function (result) {
                        if (!result.ok) {
                            console.error("calibration update failed:", result.error);
                        }
                    });
                }, 300);
            });
        });

        /* Load initial values */
        eel.get_calibration()(function (params) {
            if (!params) return;
            inputs.forEach(function (input) {
                const key = input.dataset.key;
                if (key in params) {
                    input.value = params[key];
                }
            });
        });
    }

    /* --- Main Polling Loop --- */
    async function pollSensorData() {
        try {
            const data = await eel.get_sensor_data()();
            if (!data) return;

            renderHeatmap(data.heatmap, data.rows, data.cols);
            updateWeight(data.weight_kg);
            updateStatus(data.status);
            DOM.gridSize.textContent = `${data.rows}×${data.cols}`;

            /* Read timeout from calibration for timer bar scaling */
            const calibSnap = await eel.get_calibration()();
            const timeout = calibSnap ? calibSnap.posture_timeout_seconds : 60;
            updateTimer(data.static_seconds, timeout, data.is_alert);

            /* Alert logic */
            if (data.is_alert && !lastAlertState) {
                alertDismissed = false;
                showAlert(data.static_seconds);
            }
            if (!data.is_alert) {
                alertDismissed = false;
                hideAlert();
            }
            lastAlertState = data.is_alert;

            updateFps();
        } catch (err) {
            console.error("poll error:", err);
        }
    }

    /* --- Init --- */
    function init() {
        initCalibrationPanel();
        setInterval(pollSensorData, 50); /* ~20 FPS UI updates */
    }

    /* Wait for Eel to be ready */
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
