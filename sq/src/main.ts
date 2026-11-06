// ============================================================================
// SYNAPSE-QUANTUM: Cleanroom Console UI (Node 17) & 17-node orchestrator.
// ============================================================================

import { NEURON_COUNT, FLOATS_PER_NEURON, THERMAL_LIMIT_C, RESERVOIR_BINS } from './constants';
import { HestonLiquidStateEngine } from './data-reservoir/heston_lsm';
import { generateOfflineComplianceBundle } from './nodes/ecosystem_nodes';
import { BiosecuritySandboxEngine, evaluateAdversarialVivaResponse } from './nodes/biosecurity_sandbox';

const $ = (id: string) => document.getElementById(id);
const setTxt = (id: string, val: string) => { const el = $(id); if (el) el.textContent = val; };

const hestonEngine = new HestonLiquidStateEngine();
const sandboxEngine = new BiosecuritySandboxEngine();

// 1. Telemetry buffer. SharedArrayBuffer needs cross-origin isolation (COOP/COEP headers).
const byteSize = NEURON_COUNT * FLOATS_PER_NEURON * Float32Array.BYTES_PER_ELEMENT;
const canShare = typeof SharedArrayBuffer !== 'undefined' && self.crossOriginIsolated;
const sharedBuffer: ArrayBufferLike = canShare ? new SharedArrayBuffer(byteSize) : new ArrayBuffer(byteSize);
const telemetryView = new Float32Array(sharedBuffer);

// 2. Canvas at device resolution, then hand it to the worker
const canvasEl = $('slm-canvas') as HTMLCanvasElement;
const dpr = Math.min(window.devicePixelRatio || 1, 2);
const cssSize = () => ({ w: Math.max(320, Math.round(canvasEl.clientWidth * dpr)), h: Math.round(450 * dpr) });
{ const s = cssSize(); canvasEl.width = s.w; canvasEl.height = s.h; }
const offscreen = canvasEl.transferControlToOffscreen();
const worker = new Worker(new URL('./core-physics/workers/physics.worker.ts', import.meta.url), { type: 'module' });
worker.postMessage({ type: 'INIT', payload: { sharedBuffer, canvas: offscreen } }, [offscreen]);
new ResizeObserver(() => { const s = cssSize(); worker.postMessage({ type: 'RESIZE', payload: { width: s.w, height: s.h } }); }).observe(canvasEl);

let spikeRateHz = 0;
let gsText = '...';
worker.onmessage = (e) => {
  const { type, payload } = e.data;
  if (type === 'READY') {
    const computeTag = payload.hasWebGPU ? 'WebGPU RK4' : 'CPU HH';
    const renderTag = payload.hasWebGL2 ? 'WebGL2 3D Volumetric' : '2D Fallback';
    const iso = payload.snapshots ? ' | SAB unavailable: snapshot mode (check COOP/COEP)' : '';
    setTxt('gpu-status', `${computeTag} + ${renderTag} (${payload.neuronCount} Neurons)${iso}`);
  } else if (type === 'STATS') {
    spikeRateHz = payload.spikesPerNeuronPerSec;
    gsText = `${spikeRateHz.toFixed(2)} spk/neuron/s | GS eff=${(payload.gsEfficiency * 100).toFixed(0)}% unif=${(payload.gsUniformity * 100).toFixed(0)}% [${payload.backend}]`;
  } else if (type === 'SNAPSHOT') {
    telemetryView.set(new Float32Array(e.data.buffer));
  }
};

// 3. Orbit camera + Node 11 artifact tracer
let isDragging = false, dragMoved = false, lastX = 0, lastY = 0;
let camYaw = 0.4, camPitch = 0.25, camZoom = 680.0;

canvasEl.addEventListener('pointerdown', (ev) => { isDragging = true; dragMoved = false; lastX = ev.clientX; lastY = ev.clientY; });
window.addEventListener('pointermove', (ev) => {
  if (!isDragging) return;
  const dx = ev.clientX - lastX, dy = ev.clientY - lastY;
  if (Math.abs(dx) + Math.abs(dy) > 3) dragMoved = true;
  lastX = ev.clientX; lastY = ev.clientY;
  camYaw += dx * 0.008;
  camPitch = Math.max(-1.3, Math.min(1.3, camPitch + dy * 0.008));
  worker.postMessage({ type: 'UPDATE_PARAMS', payload: { camera_yaw: camYaw, camera_pitch: camPitch } });
});
window.addEventListener('pointerup', (ev) => {
  if (isDragging && !dragMoved) {
    const rect = canvasEl.getBoundingClientRect();
    const x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -(((ev.clientY - rect.top) / rect.height) * 2 - 1);
    setTxt('bug-sniffer-log', `[Node 11 Artifact Tracer] Wavefront aberration logged at SLM coord (${x.toFixed(3)}, ${y.toFixed(3)}) -> Zernike phase correction queued.`);
  }
  isDragging = false;
});
canvasEl.addEventListener('wheel', (ev) => {
  ev.preventDefault();
  camZoom = Math.max(220.0, Math.min(1300.0, camZoom + ev.deltaY * 0.45));
  worker.postMessage({ type: 'UPDATE_PARAMS', payload: { camera_zoom: camZoom } });
}, { passive: false });

// 4. Controls
const laserSlider = $('laser-power') as HTMLInputElement;
const spreadSlider = $('market-spread') as HTMLInputElement;
const focalSlider = $('focal-depth') as HTMLInputElement;
const paletteSelect = $('palette-select') as HTMLSelectElement;
const dendriteToggle = $('dendrite-toggle') as HTMLInputElement;
const liveDrive = $('live-drive') as HTMLInputElement;

function pushParams() {
  worker.postMessage({
    type: 'UPDATE_PARAMS',
    payload: {
      p_laser: parseFloat(laserSlider.value || '35'),
      slm_focal_z: parseFloat(focalSlider.value || '0'),
      palette_id: parseFloat(paletteSelect.value || '0'),
      show_dendrites: dendriteToggle.checked ? 1 : 0,
      ...(liveDrive.checked && hestonEngine.isFeedLive() ? {} : { market_modulation: parseFloat(spreadSlider.value || '1.25') }),
    },
  });
}
[laserSlider, spreadSlider, focalSlider].forEach((el) => el.addEventListener('input', pushParams));
[paletteSelect, dendriteToggle].forEach((el) => el.addEventListener('change', pushParams));

// 5. Telemetry loop (10 Hz)
let latestSummary = { meanVoltage: '-65.0', firingRateHz: '0.0', impliedVol: '20.5%', atpLevel: '3.20', tempCelsius: '37.00',
  laserMw: '35', focalZ: '0', alphaBps: '0.00', throttled: 'false' };

let recording = false;
const recRows: string[] = [];
const recBtn = $('record-btn') as HTMLButtonElement;
const REC_HEADER = ['t_ms', ...Array.from({ length: RESERVOIR_BINS }, (_, i) => `s${i}`), 'mid', 'spread_bps', 'imbalance'].join(',');
const MAX_REC_ROWS = 36000; // one hour at 10 Hz

function downloadText(name: string, text: string, mime = 'text/csv') {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const a = document.createElement('a');
  a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

recBtn.addEventListener('click', () => {
  if (!recording) {
    if (!hestonEngine.isFeedLive()) { setTxt('bug-sniffer-log', '[Recorder] Live L2 feed is not connected. Recording synthetic prices would produce useless training data.'); return; }
    recRows.length = 0; recording = true; recBtn.textContent = 'Stop & download CSV';
  } else {
    recording = false; recBtn.textContent = 'Record reservoir states + L2 features (CSV for training)';
    downloadText('reservoir_recording.csv', [REC_HEADER, ...recRows].join('\n'));
  }
});

setInterval(() => {
  const nowMs = performance.now();
  const laserMw = parseFloat(laserSlider.value || '35');
  const h = hestonEngine.evaluateTick(telemetryView, NEURON_COUNT, FLOATS_PER_NEURON, laserMw * 0.001, nowMs);

  worker.postMessage({ type: 'THROTTLE_LEDGER', payload: { throttle: h.throttled } });
  if (liveDrive.checked && h.feedLive) {
    worker.postMessage({ type: 'UPDATE_PARAMS', payload: { market_modulation: h.spreadVelocity, market_imbalance: h.imbalance } });
    spreadSlider.value = h.spreadVelocity.toFixed(2);
    spreadSlider.disabled = true;
  } else {
    spreadSlider.disabled = false;
    worker.postMessage({ type: 'UPDATE_PARAMS', payload: { market_imbalance: 0 } });
  }

  if (recording && recRows.length < MAX_REC_ROWS && h.feedLive) {
    recRows.push([nowMs.toFixed(0), ...Array.from(h.reservoirState, (v) => v.toFixed(5)), h.spotPrice.toFixed(2), h.bidAskSpreadBps.toFixed(4), h.imbalance.toFixed(4)].join(','));
    if (recRows.length >= MAX_REC_ROWS) recBtn.click();
  }

  latestSummary = {
    meanVoltage: h.meanVm.toFixed(2), firingRateHz: spikeRateHz.toFixed(2),
    impliedVol: `${(h.impliedVolatility * 100).toFixed(2)}%`, atpLevel: h.meanAtp.toFixed(3),
    tempCelsius: h.maxTempC.toFixed(2), laserMw: laserMw.toFixed(0), focalZ: focalSlider.value,
    alphaBps: h.predictedAlphaBps.toFixed(2), throttled: h.throttled ? h.throttleReason : 'false',
  };

  setTxt('val-vm', `${latestSummary.meanVoltage} mV`);
  setTxt('val-atp', `${latestSummary.atpLevel} mM`);
  setTxt('val-temp', `${latestSummary.tempCelsius} °C`);
  setTxt('val-vol', `${latestSummary.impliedVol} (σv=${h.volOfVol.toFixed(2)}${h.feller ? ', Feller ✓' : ', Feller ✗'})`);
  setTxt('val-spot', `$${h.spotPrice.toFixed(2)}${h.feedLive ? '' : ' (synthetic)'}`);
  setTxt('val-alpha', `${h.predictedAlphaBps.toFixed(2)} bps vs ${h.ledgerCostBps.toFixed(2)} bps${h.throttled ? ` [THROTTLED: ${h.throttleReason}]` : ''}`);
  setTxt('val-onnx', h.onnxStatus);
  setTxt('val-l2', `${h.l2FeedStatus} (Spread=${h.bidAskSpreadBps.toFixed(2)} bps, OBI=${h.imbalance.toFixed(2)})`);
  setTxt('val-upi', h.upiSettlementUri);
  setTxt('val-gs', gsText);
  setTxt('val-sentinel', h.maxTempC > THERMAL_LIMIT_C ? `ALERT: PHOTOTHERMAL THRESHOLD BREACH (${h.maxTempC.toFixed(2)} °C > ${THERMAL_LIMIT_C})` : `NOMINAL (<${THERMAL_LIMIT_C} °C)`);
}, 100);

// 6. PDF export
$('export-pdf-btn')?.addEventListener('click', async () => {
  const pdfBytes = await generateOfflineComplianceBundle(latestSummary);
  const url = URL.createObjectURL(new Blob([pdfBytes as BlobPart], { type: 'application/pdf' }));
  const a = document.createElement('a');
  a.href = url; a.download = 'SYNAPSE-QUANTUM_SIMULATED_SOP_Dossier.pdf'; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
});

// 7. Node 4 sandbox
const startSandbox = () => sandboxEngine.startSession(
  (sec) => setTxt('sandbox-timer', `${sec}s`),
  () => {
    setTxt('sandbox-prompt', '[Node 4 Sandbox Timer Expired] Click Reset to re-certify.');
    const o = $('sandbox-options'); if (o) o.innerHTML = '';
  },
);

function renderSandboxQuestion() {
  const q = sandboxEngine.getCurrentQuestion();
  const promptEl = $('sandbox-prompt'), optsEl = $('sandbox-options');
  if (!promptEl || !optsEl) return;
  promptEl.textContent = `[${q.id} // ${q.category}] ${q.prompt}`;
  optsEl.innerHTML = '';
  q.options.forEach((opt, idx) => {
    const btn = document.createElement('button');
    btn.className = 'quiz-btn';
    btn.textContent = opt;
    btn.addEventListener('click', () => {
      const res = sandboxEngine.submitAnswer(idx);
      setTxt('sandbox-feedback', `${res.correct ? 'PASS' : 'FAIL'} (${res.score}/${res.answered}) — ${res.explanation}`);
      if (!res.completed) renderSandboxQuestion();
      else { promptEl.textContent = `[Node 4 Biosecurity Certification Complete] Final Score: ${res.score}/${res.total}`; optsEl.innerHTML = ''; }
    });
    optsEl.appendChild(btn);
  });
}

startSandbox();
renderSandboxQuestion();
$('sandbox-reset-btn')?.addEventListener('click', () => { startSandbox(); renderSandboxQuestion(); setTxt('sandbox-feedback', 'New randomised question set loaded.'); });

// 8. Node 16 viva
$('viva-submit-btn')?.addEventListener('click', () => {
  const input = $('viva-input') as HTMLTextAreaElement | null;
  if (!input) return;
  const res = evaluateAdversarialVivaResponse(input.value);
  setTxt('viva-feedback', `${res.verdict} — ${res.critique}`);
});
