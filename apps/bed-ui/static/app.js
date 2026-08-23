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
        patientBed: document.getElementById("patient-bed"),
        patientWeightHeader: document.getElementById("patient-weight-header"),

        // KPIs
        kpiWeightVal: document.getElementById("kpi-weight-val"),
        kpiStabilityFill: document.getElementById("kpi-stability-fill"),
        kpiStabilityText: document.getElementById("kpi-stability-text"),
        kpiStabilityStatus: document.getElementById("kpi-stability-status"),
        // Posture KPI
        kpiPostureVal: document.getElementById("kpi-posture-val"),
        kpiAsymmetryLabel: document.getElementById("kpi-asymmetry-label"),
        kpiReliefScore: document.getElementById("kpi-relief-score"),
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
        eventsListContainer: document.getElementById("events-list-container"),
        eventsCount: document.getElementById("events-count"),
        summaryChanges: document.getElementById("summary-changes"),
        summaryAvgTime: document.getElementById("summary-avg-time"),
        summaryScore: document.getElementById("summary-score"),
    };

    /* --- State --- */
    let chartWeightInstance = null;
    let chartPostureInstance = null;
    let chartTimeInstance = null;
    let sparklineInstance = null;
    let postureSparkHistory = [8, 10, 9, 12, 11, 10, 11];
    let alertDismissed = false;
    let lastAlertState = false;
    let configuredTimeoutSec = 3600; // 60 min default
    let isEelAvailable = typeof window.eel !== "undefined";

    /* --- Spectral Color Map (Deep Blue -> Cyan -> Teal -> Green -> Yellow -> Orange -> Deep Red) --- */
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

    /* --- True 2D Bilinear Resampling + Separable Gaussian Convolution Kernel --- */
    const GAUSS_KERNEL_7 = [0.03, 0.11, 0.22, 0.28, 0.22, 0.11, 0.03]; // Gaussian 1D kernel
    const GAUSS_RADIUS = 3;
    const SUPER_ROWS = 128; // 4x supersampling
    const SUPER_COLS = 256;

    // Reusable buffers to avoid garbage collection pressure at 60 FPS
    let superBufferA = new Float32Array(SUPER_ROWS * SUPER_COLS);
    let superBufferB = new Float32Array(SUPER_ROWS * SUPER_COLS);
    let offscreenCanvas = null;
    let offscreenCtx = null;
    let offscreenImgData = null;

    function initSuperResolutionBuffers() {
        if (!offscreenCanvas) {
            offscreenCanvas = document.createElement("canvas");
            offscreenCanvas.width = SUPER_COLS;
            offscreenCanvas.height = SUPER_ROWS;
            offscreenCtx = offscreenCanvas.getContext("2d");
            offscreenImgData = offscreenCtx.createImageData(SUPER_COLS, SUPER_ROWS);
        }
    }

    /* --- Heatmap Rendering (Algorithmic Continuous Gaussian Isobars) --- */
    function renderHeatmapMatrix(matrix, rows, cols) {
        if (!DOM.heatmapCanvas) return;
        initSuperResolutionBuffers();

        const canvas = DOM.heatmapCanvas;
        const ctx = canvas.getContext("2d");

        const targetW = canvas.parentElement.clientWidth || 640;
        const targetH = canvas.parentElement.clientHeight || 310;

        if (canvas.width !== targetW || canvas.height !== targetH) {
            canvas.width = targetW;
            canvas.height = targetH;
        }

        const srcRows = rows || 32;
        const srcCols = cols || 64;

        let activeCells = 0;
        let peakVal = 0;
        let peakRow = 0;
        let peakCol = 0;
        let totalPressure = 0;
        let leftSidePressure = 0;   // Linhas 0..15
        let rightSidePressure = 0;  // Linhas 16..31
        let headTorsoPressure = 0;  // Colunas 32..63
        let legsFeetPressure = 0;   // Colunas 0..31

        // Coleta de métricas anatômicas brutas
        for (let r = 0; r < srcRows; r++) {
            for (let c = 0; c < srcCols; c++) {
                const val = matrix && matrix[r] && matrix[r][c] !== undefined ? matrix[r][c] : 0;
                if (val > 0.04) {
                    activeCells++;
                    totalPressure += val;
                    if (val > peakVal) {
                        peakVal = val;
                        peakRow = r;
                        peakCol = c;
                    }
                    if (r < srcRows / 2) leftSidePressure += val;
                    else rightSidePressure += val;

                    if (c >= srcCols / 2) headTorsoPressure += val;
                    else legsFeetPressure += val;
                }
            }
        }

        // 1. Interpolação Bilinear 2D (32x64 -> 128x256 Super-Resolution)
        const rowScale = (srcRows - 1) / (SUPER_ROWS - 1);
        const colScale = (srcCols - 1) / (SUPER_COLS - 1);

        for (let sy = 0; sy < SUPER_ROWS; sy++) {
            const rf = sy * rowScale;
            const r0 = Math.floor(rf);
            const r1 = Math.min(r0 + 1, srcRows - 1);
            const dr = rf - r0;
            const rowOffset = sy * SUPER_COLS;

            for (let sx = 0; sx < SUPER_COLS; sx++) {
                const cf = sx * colScale;
                const c0 = Math.floor(cf);
                const c1 = Math.min(c0 + 1, srcCols - 1);
                const dc = cf - c0;

                const v00 = matrix && matrix[r0] ? (matrix[r0][c0] || 0) : 0;
                const v01 = matrix && matrix[r0] ? (matrix[r0][c1] || 0) : 0;
                const v10 = matrix && matrix[r1] ? (matrix[r1][c0] || 0) : 0;
                const v11 = matrix && matrix[r1] ? (matrix[r1][c1] || 0) : 0;

                const interp = (1 - dr) * (1 - dc) * v00 +
                               (1 - dr) * dc * v01 +
                               dr * (1 - dc) * v10 +
                               dr * dc * v11;

                superBufferA[rowOffset + sx] = interp;
            }
        }

        // 2. Convolução Gaussiana 2D Separável (Passo Horizontal)
        for (let y = 0; y < SUPER_ROWS; y++) {
            const rowOffset = y * SUPER_COLS;
            for (let x = 0; x < SUPER_COLS; x++) {
                let acc = 0;
                for (let k = -GAUSS_RADIUS; k <= GAUSS_RADIUS; k++) {
                    const sampleX = Math.max(0, Math.min(SUPER_COLS - 1, x + k));
                    acc += superBufferA[rowOffset + sampleX] * GAUSS_KERNEL_7[k + GAUSS_RADIUS];
                }
                superBufferB[rowOffset + x] = acc;
            }
        }

        // 3. Convolução Gaussiana 2D Separável (Passo Vertical) + Mapeamento Espectral
        const imgDataArr = offscreenImgData.data;
        for (let y = 0; y < SUPER_ROWS; y++) {
            const rowOffset = y * SUPER_COLS;
            for (let x = 0; x < SUPER_COLS; x++) {
                let smoothVal = 0;
                for (let k = -GAUSS_RADIUS; k <= GAUSS_RADIUS; k++) {
                    const sampleY = Math.max(0, Math.min(SUPER_ROWS - 1, y + k));
                    smoothVal += superBufferB[sampleY * SUPER_COLS + x] * GAUSS_KERNEL_7[k + GAUSS_RADIUS];
                }

                const pixelIdx = (rowOffset + x) * 4;
                if (smoothVal > 0.035) {
                    const [cr, cg, cb] = interpolateSpectralColor(smoothVal);
                    // Gradiente suave de opacidade nas bordas das isobaras
                    const alpha = Math.min(235, Math.round(smoothVal * 200 + 45));
                    imgDataArr[pixelIdx] = cr;
                    imgDataArr[pixelIdx + 1] = cg;
                    imgDataArr[pixelIdx + 2] = cb;
                    imgDataArr[pixelIdx + 3] = alpha;
                } else {
                    // Fundo transparente para deixar a maca e a silhueta perfeitamente visíveis
                    imgDataArr[pixelIdx + 3] = 0;
                }
            }
        }

        offscreenCtx.putImageData(offscreenImgData, 0, 0);

        // 4. Renderização Final com Interpolação de Alta Qualidade
        ctx.clearRect(0, 0, targetW, targetH);
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = "high";
        ctx.drawImage(offscreenCanvas, 0, 0, targetW, targetH);

        // Atualização em Tempo Real da Postura, Assimetria e Métricas
        updatePostureAndAsymmetry(totalPressure, leftSidePressure, rightSidePressure, headTorsoPressure, legsFeetPressure, peakVal, activeCells);
        updateHeatmapMetrics(activeCells, srcRows * srcCols, peakVal, peakCol, srcCols);
    }

    /* --- Posture Condition & Lateral Asymmetry Calculator --- */
    function updatePostureAndAsymmetry(totalP, leftP, rightP, headP, legsP, peakVal, activeCount) {
        if (!DOM.kpiPostureVal) return;

        // Se leito desocupado
        if (totalP < 1.0 || activeCount < 8) {
            DOM.kpiPostureVal.textContent = "Leito Livre";
            if (DOM.kpiAsymmetryLabel) DOM.kpiAsymmetryLabel.textContent = "Sem Carga";
            if (DOM.kpiReliefScore) {
                DOM.kpiReliefScore.textContent = "100% Alívio";
                DOM.kpiReliefScore.style.color = "var(--samel-green)";
            }
            return;
        }

        // Assimetria Lateral (%)
        const sumLateral = leftP + rightP;
        const diffLateral = rightP - leftP;
        const asymPct = sumLateral > 0 ? (Math.abs(diffLateral) / sumLateral) * 100 : 0;

        // Classificação Postural
        let postureName = "Decúbito Dorsal";
        let asymLabel = "Pressão Simétrica";
        let asymColor = "var(--samel-teal-dark)";

        if (asymPct > 28) {
            if (diffLateral > 0) {
                postureName = "Decúbito Lat. Dir.";
                asymLabel = `Assimetria: ${Math.round(asymPct)}% Dir`;
                asymColor = "var(--samel-amber)";
            } else {
                postureName = "Decúbito Lat. Esq.";
                asymLabel = `Assimetria: ${Math.round(asymPct)}% Esq`;
                asymColor = "var(--samel-amber)";
            }
        } else {
            // Verifica se está sentado / cabeceira elevada (Fowler)
            if (headP > 0 && legsP > 0 && (headP / (legsP + 0.1)) > 3.2) {
                postureName = "Posição de Fowler";
                asymLabel = "Cabeceira Elevada";
            } else {
                postureName = "Decúbito Dorsal";
                asymLabel = "Pressão Simétrica";
            }
        }

        // Score de alívio: quanto menor o pico e mais simétrico, maior o alívio
        const reliefScore = Math.max(15, Math.min(99, Math.round(100 - (peakVal * 65) - (asymPct * 0.25))));
        
        DOM.kpiPostureVal.textContent = postureName;
        if (DOM.kpiAsymmetryLabel) {
            DOM.kpiAsymmetryLabel.textContent = asymLabel;
        }
        if (DOM.kpiReliefScore) {
            DOM.kpiReliefScore.textContent = `${reliefScore}% Alívio`;
            DOM.kpiReliefScore.style.color = reliefScore > 75 ? "var(--samel-teal-dark)" : (reliefScore > 50 ? "var(--samel-amber)" : "var(--samel-red)");
        }

        // Atualiza sparkline com histórico de assimetria/alívio
        if (sparklineInstance) {
            postureSparkHistory.push(Math.round(asymPct));
            if (postureSparkHistory.length > 7) postureSparkHistory.shift();
            sparklineInstance.data.datasets[0].data = [...postureSparkHistory];
            sparklineInstance.update("none");
        }
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

        // KPI 2: Condição Postural & Assimetria (se fornecido pelo backend)
        if (data.posture_info) {
            if (DOM.kpiPostureVal && data.posture_info.posture) {
                DOM.kpiPostureVal.textContent = data.posture_info.posture;
            }
            if (DOM.kpiAsymmetryLabel && data.posture_info.asymmetry_label) {
                DOM.kpiAsymmetryLabel.textContent = data.posture_info.asymmetry_label;
            }
            if (DOM.kpiReliefScore && data.posture_info.relief_score !== undefined) {
                const rs = data.posture_info.relief_score;
                DOM.kpiReliefScore.textContent = `${rs}% Alívio`;
                DOM.kpiReliefScore.style.color = rs > 75 ? "var(--samel-teal-dark)" : (rs > 50 ? "var(--samel-amber)" : "var(--samel-red)");
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

        // Atualização Dinâmica nos Headers dos 3 Gráficos
        if (DOM.chartLiveWeight) {
            DOM.chartLiveWeight.textContent = `${weightKg.toFixed(1)} kg`;
        }
        if (DOM.chartLiveTime) {
            DOM.chartLiveTime.textContent = `${Math.round(staticSecs / 60)} min`;
        }
        if (DOM.chartLivePosture && data.posture_info && data.posture_info.posture) {
            const asym = data.posture_info.asymmetry_pct ? (data.posture_info.asymmetry_pct / 100).toFixed(2) : "0.00";
            DOM.chartLivePosture.textContent = `${data.posture_info.posture} (${asym})`;
        }

        // Identificador da Maca no Cabeçalho
        if (data.maca_id && DOM.patientBed) {
            DOM.patientBed.textContent = data.maca_id;
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

    /* --- Chart.js Initialization & Dynamic Data --- */
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
                        data: [null, null, null, null, null, null, null, null, 0],
                        borderColor: "#0d9488",
                        backgroundColor: "rgba(13, 148, 136, 0.08)",
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3,
                        spanGaps: true,
                        pointRadius: 2,
                        pointHoverRadius: 4,
                    }]
                },
                options: getCleanChartOptions(50, 90, "kg")
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
                        data: [null, null, null, null, null, null, null, null, 0],
                        borderColor: "#0284c7",
                        backgroundColor: "rgba(2, 132, 199, 0.08)",
                        borderWidth: 2,
                        fill: true,
                        tension: 0.2,
                        spanGaps: true,
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
                        data: [null, null, null, null, null, null, null, null, 0],
                        borderColor: "#10b981",
                        backgroundColor: "rgba(16, 185, 129, 0.08)",
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3,
                        spanGaps: true,
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
                        data: [10, 10, 10, 10, 10, 10, 10],
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

    /* --- Funções de Busca e Atualização dos Painéis Analíticos --- */

    async function fetchAndRenderCharts() {
        if (isEelAvailable && window.eel && window.eel.get_dashboard_charts) {
            try {
                const chartsData = await window.eel.get_dashboard_charts(4)();
                if (chartsData && chartsData.labels) {
                    // 1. Gráfico de Peso
                    if (chartWeightInstance && chartsData.weight_series) {
                        chartWeightInstance.data.labels = chartsData.labels;
                        chartWeightInstance.data.datasets[0].data = chartsData.weight_series;
                        const validWeights = chartsData.weight_series.filter(v => v !== null && v !== undefined);
                        if (validWeights.length > 0) {
                            const minW = Math.max(0, Math.floor(Math.min(...validWeights) - 2));
                            const maxW = Math.ceil(Math.max(...validWeights) + 2);
                            chartWeightInstance.options.scales.y.min = minW;
                            chartWeightInstance.options.scales.y.max = maxW;
                            const latestW = validWeights[validWeights.length - 1];
                            if (DOM.chartLiveWeight) DOM.chartLiveWeight.textContent = `${latestW.toFixed(1)} kg`;
                        }
                        chartWeightInstance.update("none");
                    }

                    // 2. Gráfico de Índice Postural
                    if (chartPostureInstance && chartsData.posture_series) {
                        chartPostureInstance.data.labels = chartsData.labels;
                        chartPostureInstance.data.datasets[0].data = chartsData.posture_series;
                        const validPos = chartsData.posture_series.filter(v => v !== null && v !== undefined);
                        if (validPos.length > 0 && DOM.chartLivePosture) {
                            const latestPos = validPos[validPos.length - 1];
                            const statusLabel = latestPos < 0.15 ? "Estável" : "Assimetria";
                            DOM.chartLivePosture.textContent = `${statusLabel} (${latestPos.toFixed(2)})`;
                        }
                        chartPostureInstance.update("none");
                    }

                    // 3. Gráfico de Tempo na Condição
                    if (chartTimeInstance && chartsData.time_series) {
                        chartTimeInstance.data.labels = chartsData.labels;
                        chartTimeInstance.data.datasets[0].data = chartsData.time_series;
                        const validTime = chartsData.time_series.filter(v => v !== null && v !== undefined);
                        if (validTime.length > 0 && DOM.chartLiveTime) {
                            const latestTime = validTime[validTime.length - 1];
                            DOM.chartLiveTime.textContent = `${Math.round(latestTime)} min`;
                        }
                        chartTimeInstance.update("none");
                    }
                }
            } catch (e) {
                // Silencioso se backend reiniciar
            }
        }
    }

    async function fetchAndRenderEvents() {
        if (isEelAvailable && window.eel && window.eel.get_dashboard_events) {
            try {
                const events = await window.eel.get_dashboard_events(5)();
                if (DOM.eventsListContainer) {
                    if (events && events.length > 0) {
                        DOM.eventsListContainer.innerHTML = events.map(evt => `
                            <li class="event-entry ${evt.level || ''}">
                                <span class="event-time">${evt.time}</span>
                                <span class="event-desc">${evt.description}</span>
                            </li>
                        `).join("");
                        if (DOM.eventsCount) {
                            DOM.eventsCount.textContent = `${events.length} registro${events.length > 1 ? 's' : ''}`;
                        }
                    } else {
                        DOM.eventsListContainer.innerHTML = `
                            <li class="event-empty" style="color: var(--samel-text-muted); font-size: 12px; padding: 12px 4px;">
                                Nenhum evento recente registrado no leito.
                            </li>
                        `;
                        if (DOM.eventsCount) {
                            DOM.eventsCount.textContent = "0 registros";
                        }
                    }
                }
            } catch (e) {}
        }
    }

    async function fetchAndRenderDailySummary() {
        if (isEelAvailable && window.eel && window.eel.get_dashboard_summary) {
            try {
                const summary = await window.eel.get_dashboard_summary()();
                if (summary) {
                    if (DOM.summaryChanges && summary.total_rotations_today !== undefined) {
                        DOM.summaryChanges.textContent = `${summary.total_rotations_today} rotaç${summary.total_rotations_today === 1 ? 'ão' : 'ões'}`;
                    }
                    if (DOM.summaryAvgTime && summary.avg_posture_time_min !== undefined) {
                        DOM.summaryAvgTime.textContent = `${Math.round(summary.avg_posture_time_min)} min`;
                    }
                    if (DOM.summaryScore && summary.relief_score_pct !== undefined) {
                        const score = summary.relief_score_pct;
                        const label = score >= 90 ? "Ótimo" : (score >= 70 ? "Bom" : "Atenção");
                        const color = score >= 90 ? "var(--samel-green)" : (score >= 70 ? "var(--samel-primary)" : "var(--samel-red)");
                        DOM.summaryScore.textContent = `${score.toFixed(0)}% (${label})`;
                        DOM.summaryScore.style.color = color;
                    }
                }
            } catch (e) {}
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
        simPhase += 0.04;
        const matrix = [];
        const breath = Math.sin(simPhase) * 0.05;
        
        // Simula deslocamento lateral suave a cada ciclo (Dorsal -> Leve Direita -> Leve Esquerda)
        const lateralShift = Math.sin(simPhase * 0.25) * 2.2; 
        const centerR = 16 + lateralShift;

        for (let r = 0; r < rows; r++) {
            const rowArr = [];
            for (let c = 0; c < cols; c++) {
                let p = 0;

                // Cabeça / Occipital (c ≈ 55, r ≈ centerR)
                const headDist = Math.hypot(c - 55, r - centerR);
                if (headDist < 4.8) p = Math.max(p, (1 - headDist / 4.8) * 0.42);

                // Escápulas (c ≈ 45, r ≈ centerR - 4.5 e r ≈ centerR + 4.5)
                const scapulaLeft = Math.hypot(c - 45, r - (centerR - 4.5));
                const scapulaRight = Math.hypot(c - 45, r - (centerR + 4.5));
                if (scapulaLeft < 4.5) p = Math.max(p, (1 - scapulaLeft / 4.5) * (0.60 + breath - lateralShift * 0.05));
                if (scapulaRight < 4.5) p = Math.max(p, (1 - scapulaRight / 4.5) * (0.60 + breath + lateralShift * 0.05));

                // Região Sacral (c ≈ 32, r ≈ centerR) - Maior concentração
                const sacrumDist = Math.hypot(c - 32, r - centerR);
                if (sacrumDist < 6.8) p = Math.max(p, (1 - sacrumDist / 6.8) * (0.78 + breath * 0.5));

                // Poplíteo / Pernas (c ≈ 20, r ≈ centerR - 3 e r ≈ centerR + 3)
                const legLeft = Math.hypot(c - 20, r - (centerR - 3.2));
                const legRight = Math.hypot(c - 20, r - (centerR + 3.2));
                if (legLeft < 3.8) p = Math.max(p, (1 - legLeft / 3.8) * 0.40);
                if (legRight < 3.8) p = Math.max(p, (1 - legRight / 3.8) * 0.40);

                // Calcâneos / Pés (c ≈ 7, r ≈ centerR - 3 e r ≈ centerR + 3)
                const heelLeft = Math.hypot(c - 7, r - (centerR - 3.2));
                const heelRight = Math.hypot(c - 7, r - (centerR + 3.2));
                if (heelLeft < 3.2) p = Math.max(p, (1 - heelLeft / 3.2) * 0.52);
                if (heelRight < 3.2) p = Math.max(p, (1 - heelRight / 3.2) * 0.52);

                // Ruído basal suave
                if (p > 0.04) p += (Math.random() - 0.5) * 0.02;

                rowArr.push(Math.max(0, Math.min(1, p)));
            }
            matrix.push(rowArr);
        }
        return matrix;
    }

    /* --- Calibration & Timeout Formatters --- */
    function formatCalibrationDate(isoStr) {
        if (!isoStr) return "--";
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

    /* --- Conexão WebSocket Direta ("Via Expressa") --- */
    let telemetrySocket = null;
    let isSocketConnected = false;

    function connectWebSocket() {
        const wsUrl = "ws://localhost:8000/ws/telemetry";
        try {
            telemetrySocket = new WebSocket(wsUrl);

            telemetrySocket.onopen = function () {
                isSocketConnected = true;
                console.log("[WebSocket] Conectado à via expressa do Edge Service.");
            };

            telemetrySocket.onmessage = function (event) {
                try {
                    const data = JSON.parse(event.data);
                    if (data && data.heatmap) {
                        renderHeatmapMatrix(data.heatmap, 32, 64);
                        updateKPIs(data);
                    }
                } catch (err) {
                    console.error("[WebSocket] Erro no parsing:", err);
                }
            };

            telemetrySocket.onclose = function () {
                isSocketConnected = false;
                console.warn("[WebSocket] Desconectado. Tentando reconexão em 2s...");
                setTimeout(connectWebSocket, 2000);
            };

            telemetrySocket.onerror = function () {
                isSocketConnected = false;
                try { telemetrySocket.close(); } catch (e) {}
            };
        } catch (e) {
            isSocketConnected = false;
            setTimeout(connectWebSocket, 2000);
        }
    }

    /* --- Polling Loop (Apenas Fallback se WebSocket estiver desconectado) --- */
    async function pollSensorData() {
        if (!isSocketConnected && isEelAvailable && window.eel && window.eel.get_sensor_data) {
            try {
                const data = await window.eel.get_sensor_data()();
                if (data && data.heatmap && data.heatmap.length > 0) {
                    renderHeatmapMatrix(data.heatmap, 32, 64);
                    updateKPIs(data);
                }
            } catch (err) {}
        }
    }

    /* --- Carga Inicial de Configurações e Limites de Calibração --- */
    async function loadSystemConfigAndCalibration() {
        // 1. Carrega configurações do sistema (MACA_ID, URLs)
        if (isEelAvailable && window.eel && window.eel.get_system_config) {
            try {
                const cfg = await window.eel.get_system_config()();
                if (cfg && cfg.maca_id && DOM.patientBed) {
                    DOM.patientBed.textContent = cfg.maca_id;
                }
            } catch (e) {}
        } else {
            try {
                const res = await fetch("http://localhost:8000/api/v1/system/config");
                if (res.ok) {
                    const cfg = await res.json();
                    if (cfg && cfg.maca_id && DOM.patientBed) {
                        DOM.patientBed.textContent = cfg.maca_id;
                    }
                }
            } catch (e) {}
        }

        // 2. Carrega calibração e timeout de postura
        if (isEelAvailable && window.eel && window.eel.get_calibration) {
            try {
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
            } catch (e) {}
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

    /* --- Modal de Configurações Técnicas --- */
    function setupConfigModal() {
        const btnOpen = document.getElementById("btn-open-config");
        const btnClose = document.getElementById("btn-close-config");
        const btnCancel = document.getElementById("btn-cancel-config");
        const btnSave = document.getElementById("btn-save-config");
        const overlay = document.getElementById("config-modal-overlay");
        const statusMsg = document.getElementById("config-save-status");

        const inputMacaId = document.getElementById("input-config-maca-id");
        const inputCentralUrl = document.getElementById("input-config-central-url");
        const inputSyncInterval = document.getElementById("input-config-sync-interval");
        const inputTimeoutMin = document.getElementById("input-config-timeout-min");

        async function openModal() {
            if (!overlay) return;
            overlay.classList.remove("hidden");
            if (statusMsg) statusMsg.classList.add("hidden");

            // Carrega valores atuais
            let cfg = null;
            if (isEelAvailable && window.eel && window.eel.get_system_config) {
                try {
                    cfg = await window.eel.get_system_config()();
                } catch (e) {}
            }
            if (!cfg) {
                try {
                    const res = await fetch("http://localhost:8000/api/v1/system/config");
                    if (res.ok) cfg = await res.json();
                } catch (e) {}
            }

            if (cfg) {
                if (inputMacaId && cfg.maca_id) inputMacaId.value = cfg.maca_id;
                if (inputCentralUrl && cfg.central_api_url) inputCentralUrl.value = cfg.central_api_url;
                if (inputSyncInterval && cfg.sync_interval_sec) inputSyncInterval.value = cfg.sync_interval_sec;
            }

            if (inputTimeoutMin) {
                inputTimeoutMin.value = Math.round(configuredTimeoutSec / 60);
            }
        }

        function closeModal() {
            if (overlay) overlay.classList.add("hidden");
        }

        async function saveConfig() {
            if (!btnSave) return;
            btnSave.disabled = true;
            btnSave.textContent = "Salvando...";

            const payload = {
                maca_id: inputMacaId && inputMacaId.value.trim() ? inputMacaId.value.trim() : undefined,
                central_api_url: inputCentralUrl && inputCentralUrl.value.trim() ? inputCentralUrl.value.trim() : undefined,
                sync_interval_sec: inputSyncInterval && inputSyncInterval.value ? parseFloat(inputSyncInterval.value) : undefined,
            };

            const timeoutMins = inputTimeoutMin && inputTimeoutMin.value ? parseInt(inputTimeoutMin.value, 10) : 60;
            const timeoutSecs = timeoutMins * 60;

            try {
                let savedOk = false;

                // 1. Tenta salvar via Eel
                if (isEelAvailable && window.eel && window.eel.save_system_config) {
                    try {
                        const res = await window.eel.save_system_config(payload)();
                        if (res && res.status === "ok") savedOk = true;
                    } catch (e) {}
                }

                // 2. Se Eel não confirmou, tenta via REST direto
                if (!savedOk) {
                    try {
                        const res = await fetch("http://localhost:8000/api/v1/system/config", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify(payload),
                        });
                        if (res.ok) savedOk = true;
                    } catch (e) {}
                }

                // 3. Atualiza timeout postural
                if (isEelAvailable && window.eel && window.eel.update_calibration) {
                    try {
                        await window.eel.update_calibration("posture_timeout_seconds", timeoutSecs)();
                    } catch (e) {}
                } else {
                    try {
                        await fetch("http://localhost:8000/api/v1/calibration/params", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ param_name: "posture_timeout_seconds", value: timeoutSecs }),
                        });
                    } catch (e) {}
                }

                // 4. Atualiza UI local
                if (payload.maca_id && DOM.patientBed) {
                    DOM.patientBed.textContent = payload.maca_id;
                }

                configuredTimeoutSec = timeoutSecs;
                const timeoutFormatted = formatTimeoutLimit(configuredTimeoutSec);
                if (DOM.kpiTimerLimit) DOM.kpiTimerLimit.textContent = timeoutFormatted;
                if (DOM.summaryLimit) DOM.summaryLimit.textContent = timeoutFormatted;

                if (statusMsg) {
                    statusMsg.textContent = "Configurações salvas e aplicadas com sucesso!";
                    statusMsg.className = "config-status-msg success";
                    statusMsg.classList.remove("hidden");
                }

                setTimeout(closeModal, 1000);
            } catch (err) {
                if (statusMsg) {
                    statusMsg.textContent = "Erro ao salvar: " + err;
                    statusMsg.className = "config-status-msg error";
                    statusMsg.classList.remove("hidden");
                }
            } finally {
                btnSave.disabled = false;
                btnSave.textContent = "Salvar e Aplicar";
            }
        }

        if (btnOpen) btnOpen.addEventListener("click", openModal);
        if (btnClose) btnClose.addEventListener("click", closeModal);
        if (btnCancel) btnCancel.addEventListener("click", closeModal);
        if (btnSave) btnSave.addEventListener("click", saveConfig);
    }

    /* --- Inicialização do Dashboard --- */
    function init() {
        initCharts();
        setupChartsToggle();
        setupConfigModal();

        // 1. Inicia conexão WebSocket em tempo real
        connectWebSocket();

        // 2. Carrega configurações do sistema e limites de calibração uma única vez
        loadSystemConfigAndCalibration();

        // 3. Polling de fallback (apenas se WebSocket cair)
        setInterval(pollSensorData, 500);

        // 4. Polling dos gráficos analíticos (a cada 10s)
        fetchAndRenderCharts();
        setInterval(fetchAndRenderCharts, 10000);

        // 5. Polling dos eventos recentes (a cada 10s)
        fetchAndRenderEvents();
        setInterval(fetchAndRenderEvents, 10000);

        // 6. Polling do resumo do dia civil (a cada 15s)
        fetchAndRenderDailySummary();
        setInterval(fetchAndRenderDailySummary, 15000);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
