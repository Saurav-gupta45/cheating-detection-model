/**
 * ProctorAI Frontend v10
 * ─────────────────────────────────────────
 * - SSE: /status_feed for real-time telemetry
 * - Gauge: animated SVG suspicion ring
 * - Anomaly panel: shows WHICH anomaly is active
 * - Anomaly banner: on-video alert overlay
 * - Feature cards with status dots
 * - Feature toggles → POST /toggle
 * - Clock, log panel, calibration phases
 */

(function () {
    'use strict';

    const $ = (s) => document.querySelector(s);
    const $$ = (s) => document.querySelectorAll(s);

    // ── DOM refs ──
    const gaugePercent   = $('#gauge-percent');
    const gaugeFill      = $('#gauge-fill');
    const gaugeRiskLabel = $('#gauge-risk-label');
    const suspChipVal    = $('#suspicion-value-top');
    const suspChip       = $('#suspicion-badge');
    const gazeTimer      = $('#gaze-timer');
    const gazeTimerVal   = $('#gaze-timer-value');
    const logBody        = $('#log-body');
    const logEmpty       = $('#log-empty');
    const clockEl        = $('#clock');

    // Anomaly panel
    const anomalyList    = $('#anomaly-list');
    const anomalyEmpty   = $('#anomaly-empty');
    const anomalyCount   = $('#anomaly-count');
    const anomalyBanner  = $('#anomaly-banner');
    const anomalyBannerIcon = $('#anomaly-banner-icon');
    const anomalyBannerText = $('#anomaly-banner-text');

    // Calibration
    const calOverlay     = $('#calibration-overlay');
    const calFill        = $('#calibration-progress-fill');
    const calPct         = $('#calibration-percent');
    const calNote        = $('#calibration-note');
    const gazeCalOverlay = $('#gaze-calibration-overlay');
    const gazeCalFill    = $('#gaze-calibration-fill');
    const gazeCalPct     = $('#gaze-calibration-percent');
    const gazeStartBtn   = $('#gaze-start-btn');
    const gazeDot        = $('#gaze-dot');

    // Controls
    const btnProctor     = $('#btn-proctor');
    const btnLogsToggle  = $('#btn-logs-toggle');
    const btnLogClear    = $('#log-clear');
    const btnEnd         = $('#btn-end');

    // State
    let proctorEnabled     = true;
    let previousLogCount   = 0;
    let calPhase           = 'voice';   // 'voice' | 'gaze' | 'done'
    let gazeCalDone        = false;
    let gazeCalRunning     = false;

    const CIRCUMFERENCE = 2 * Math.PI * 52; // r=52

    // ── Clock ──
    const sessionStart = Date.now();
    function updateClock() {
        const e = Date.now() - sessionStart;
        const s = Math.floor(e / 1000) % 60;
        const m = Math.floor(e / 60000) % 60;
        const h = Math.floor(e / 3600000);
        clockEl.textContent =
            pad(h) + ':' + pad(m) + ':' + pad(s);
    }
    function pad(n) { return String(n).padStart(2, '0'); }
    setInterval(updateClock, 1000);
    updateClock();

    // ── SSE ──
    let evtSrc = null;
    function connectSSE() {
        if (evtSrc) evtSrc.close();
        evtSrc = new EventSource('/status_feed');
        evtSrc.onmessage = (e) => {
            try { updateUI(JSON.parse(e.data)); }
            catch (err) { console.error('SSE parse:', err); }
        };
        evtSrc.onerror = () => {
            evtSrc.close();
            setTimeout(connectSSE, 3000);
        };
    }
    connectSSE();

    // ── Gaze Calibration Sequence ──
    const GAZE_POS = [
        { x: 50, y: 50 }, { x: 15, y: 20 }, { x: 85, y: 20 },
        { x: 85, y: 80 }, { x: 15, y: 80 }, { x: 50, y: 50 }
    ];
    const GAZE_HOLD = 1500;

    function startGazeCal() {
        gazeCalRunning = true;
        gazeStartBtn.style.display = 'none';
        gazeDot.style.display = 'block';

        const total = GAZE_POS.length * GAZE_HOLD;
        let step = 0;
        gazeDot.style.left = GAZE_POS[0].x + '%';
        gazeDot.style.top  = GAZE_POS[0].y + '%';

        const t0 = Date.now();
        const prog = setInterval(() => {
            const pct = Math.min(100, Math.round(((Date.now() - t0) / total) * 100));
            gazeCalFill.style.width = pct + '%';
            gazeCalPct.textContent  = pct + '%';
        }, 100);

        function next() {
            step++;
            if (step >= GAZE_POS.length) {
                clearInterval(prog);
                gazeCalFill.style.width = '100%';
                gazeCalPct.textContent  = '100%';
                gazeDot.style.display   = 'none';
                gazeCalDone = true; gazeCalRunning = false;
                calPhase = 'done';
                gazeCalOverlay.classList.add('hidden');
                return;
            }
            gazeDot.style.left = GAZE_POS[step].x + '%';
            gazeDot.style.top  = GAZE_POS[step].y + '%';
            setTimeout(next, GAZE_HOLD);
        }
        setTimeout(next, GAZE_HOLD);
    }
    gazeStartBtn.addEventListener('click', startGazeCal);

    // ══════════════════════════════════════
    //  MAIN UI UPDATE  (called on each SSE)
    // ══════════════════════════════════════
    function updateUI(state) {

        // ── Phase 1: Voice Calibration ──
        if (calPhase === 'voice') {
            if (state.audio_calibrated === false) {
                calOverlay.classList.remove('hidden');
                gazeCalOverlay.classList.add('hidden');
                const pct = Math.round((state.calibration_progress || 0) * 100);
                calFill.style.width = pct + '%';
                calPct.textContent  = pct + '%';
                calNote.innerHTML = state.is_recording_speech
                    ? '<span style="color:#22c55e;font-weight:600;">🎙️ Recording voice... Keep reading.</span>'
                    : '<span style="color:#9090b0;">Waiting for speech... Read the text aloud.</span>';
                return;
            } else {
                calOverlay.classList.add('hidden');
                calPhase = 'gaze';
                gazeCalOverlay.classList.remove('hidden');
                return;
            }
        }

        // ── Phase 2: Gaze Calibration ──
        if (calPhase === 'gaze') {
            if (!gazeCalDone) return;
            calPhase = 'done';
        }

        // ── Phase 3: Running ──
        calOverlay.classList.add('hidden');
        gazeCalOverlay.classList.add('hidden');

        // ── Suspicion Gauge ──
        const susp  = Math.min(100, Math.max(0, state.suspicion || 0));
        const level = susp < 30 ? 'low' : susp < 60 ? 'medium' : 'high';
        const labels = { low: 'Safe', medium: 'Caution', high: 'At Risk' };

        gaugePercent.textContent = Math.round(susp);
        gaugePercent.className   = 'gauge-num level-' + level;
        gaugeRiskLabel.textContent = labels[level];

        const offset = CIRCUMFERENCE - (susp / 100) * CIRCUMFERENCE;
        gaugeFill.style.strokeDashoffset = offset;
        gaugeFill.className = 'gauge-arc level-' + level;

        suspChipVal.textContent = Math.round(susp) + '%';
        suspChip.className = 'suspicion-chip level-' + level;

        // ── Gaze timer ──
        const gazeOff = state.gaze_off_duration || 0;
        gazeTimer.style.display = gazeOff > 0.5 ? 'inline-flex' : 'none';
        if (gazeOff > 0.5) gazeTimerVal.textContent = gazeOff.toFixed(1);

        // ── Feature Cards + Active Anomaly tracking ──
        const activeAnomalies = [];

        if (state.feature_status) {
            Object.entries(state.feature_status).forEach(([key, fStatus]) => {
                const card   = $('#card-' + key);
                const dot    = $('#dot-' + key);
                const detail = $('#detail-' + key);
                if (!card) return;

                const st = fStatus.status; // 'OK' | 'WARNING' | 'ALERT' | 'CALIBRATING'

                // Status dot
                if (dot) {
                    dot.className = 'feat-status-dot ' +
                        (st === 'OK' ? 'st-ok' :
                         st === 'WARNING' ? 'st-warning' :
                         st === 'ALERT' ? 'st-alert' :
                         st === 'CALIBRATING' ? 'st-calibrating' : 'st-ok');
                }

                // Card background
                card.className = 'feat-card ' +
                    (st === 'ALERT' ? 'st-alert' :
                     st === 'WARNING' ? 'st-warning' :
                     st === 'CALIBRATING' ? 'st-calibrating' : '');

                // Detail text
                if (detail) detail.textContent = fStatus.detail || st;

                // Collect active anomalies
                if (st === 'ALERT' || st === 'WARNING') {
                    activeAnomalies.push({
                        key, status: st,
                        icon: FEATURE_ICONS[key] || '⚠️',
                        label: FEATURE_LABELS[key] || key,
                        detail: fStatus.detail || ''
                    });
                }
            });
        }

        // ── Render Active Anomalies Panel ──
        renderAnomalyPanel(activeAnomalies);

        // ── Warning Logs ──
        if (state.warning_logs && state.warning_logs.length > 0) {
            if (logEmpty) logEmpty.style.display = 'none';
            if (state.warning_logs.length !== previousLogCount) {
                const newLogs = state.warning_logs.slice(previousLogCount);
                newLogs.forEach(log => {
                    const entry = document.createElement('div');
                    const isAlert = log.type !== 'sound'; // heuristic
                    entry.className = 'log-entry ' + (
                        log.type === 'phone' || log.type === 'multiface' || log.type === 'unknown_speaker'
                            ? 'type-alert' : 'type-warning'
                    );
                    entry.innerHTML =
                        `<span class="log-time">${escHtml(log.timestamp)}</span>` +
                        `<span class="log-icon">${LOG_ICONS[log.type] || '⚠️'}</span>` +
                        `<span class="log-msg">${escHtml(log.message)}</span>`;
                    logBody.appendChild(entry);
                });
                logBody.scrollTop = logBody.scrollHeight;
                previousLogCount = state.warning_logs.length;
            }
        }

        // ── Proctoring state ──
        proctorEnabled = state.proctoring_enabled !== false;
        btnProctor.classList.toggle('active', proctorEnabled);
        btnProctor.querySelector('.ctrl-label').textContent =
            proctorEnabled ? 'Proctoring ON' : 'Proctoring OFF';

        // ── Sync toggles ──
        if (state.feature_toggles) {
            $$('.feature-toggle').forEach(tog => {
                if (tog.dataset.feature in state.feature_toggles) {
                    tog.checked = state.feature_toggles[tog.dataset.feature];
                }
            });
        }
    }

    // ── Anomaly Panel Renderer ──
    function renderAnomalyPanel(anomalies) {
        anomalyList.innerHTML = '';

        if (anomalies.length === 0) {
            // All clear
            anomalyList.appendChild(anomalyEmpty.cloneNode(true));
            anomalyCount.textContent = '0 alerts';
            anomalyCount.className   = 'anomaly-count';
            anomalyBanner.style.display = 'none';
            return;
        }

        // Show count
        anomalyCount.textContent = `${anomalies.length} alert${anomalies.length > 1 ? 's' : ''}`;
        anomalyCount.className   = 'anomaly-count has-alerts';

        // Render tags
        anomalies.forEach(a => {
            const tag = document.createElement('div');
            tag.className = 'anomaly-tag ' + (a.status === 'ALERT' ? 'alert' : 'warning');
            tag.textContent = a.icon + ' ' + a.label;
            tag.title = a.detail;
            anomalyList.appendChild(tag);
        });

        // Anomaly banner on video (show the most severe)
        const topAlert = anomalies.find(a => a.status === 'ALERT') || anomalies[0];
        anomalyBannerIcon.textContent = topAlert.icon;
        anomalyBannerText.textContent = topAlert.label + (topAlert.detail ? ' — ' + topAlert.detail : '');
        anomalyBanner.style.display = 'flex';
    }

    // ── Feature metadata ──
    const FEATURE_ICONS = {
        gaze:  '👁️', face:  '👤', phone: '📱',
        light: '💡', voice: '🎙️', sound: '🔊'
    };
    const FEATURE_LABELS = {
        gaze:  'Gaze Alert',    face:  'Face Issue',
        phone: 'Phone Detected', light: 'Bad Lighting',
        voice: 'Unknown Voice',  sound: 'Sound Alert'
    };
    const LOG_ICONS = {
        gaze: '👁️', no_face: '👤', multiface: '👥',
        phone: '📱', light: '💡', unknown_speaker: '🎙️',
        lip_sync: '🔇', sound: '🔊',
        unknown_speaker_time: '🎙️', unknown_speaker_freq: '🎙️'
    };

    // ── Toggle Handlers ──
    btnProctor.addEventListener('click', () => {
        proctorEnabled = !proctorEnabled;
        fetch('/toggle', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ feature: 'master', enabled: proctorEnabled })
        });
    });

    $$('.feature-toggle').forEach(tog => {
        tog.addEventListener('change', function () {
            fetch('/toggle', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ feature: this.dataset.feature, enabled: this.checked })
            });
        });
    });

    // ── Log Panel ──
    btnLogsToggle.addEventListener('click', () => {
        const log = $('#log-panel');
        log.classList.toggle('hidden');
    });

    btnLogClear.addEventListener('click', () => {
        logBody.innerHTML = '<div class="warn-log-empty">Logs cleared. Monitoring...</div>';
        previousLogCount = 999;
    });

    // ── End Session ──
    btnEnd.addEventListener('click', () => {
        if (confirm('End the proctoring session?')) {
            fetch('/toggle', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ feature: 'master', enabled: false })
            });
            document.body.innerHTML = `
                <div style="display:flex;align-items:center;justify-content:center;
                            height:100vh;background:#020203;color:#f1f0ff;
                            font-family:'Fira Sans',sans-serif;flex-direction:column;gap:20px;">
                    <div style="font-size:64px;">🛡️</div>
                    <h1 style="font-size:32px;font-weight:700;">Session Ended</h1>
                    <p style="color:#9090b0;font-size:15px;">ProctorAI proctoring session has been terminated.</p>
                    <a href="/login" style="margin-top:8px;padding:12px 28px;background:#6366f1;
                       color:#fff;border-radius:10px;text-decoration:none;font-weight:600;font-size:14px;">
                        Return to Login
                    </a>
                </div>`;
        }
    });

    // ── Helpers ──
    function escHtml(t) {
        const d = document.createElement('div');
        d.textContent = t;
        return d.innerHTML;
    }

})();
