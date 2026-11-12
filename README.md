# 🧠 SYNAPSE-QUANTUM v3.0 | *Cleanroom Console*

**[🚀 Launch Live Demo (Google Sites Embed)(https://sites.google.com/view/synapse-quantum-v3-0-cleanroom/full-page-embed)]**

An offline-first, in-browser **simulation** of a **10,240-neuron cortical organoid** driven by a two-photon holographic SLM, whose firing activity is read out by a ridge-regression liquid state machine to feed a Heston-style volatility filter and pricer. It is a research/teaching prototype. It is **not** a trading system, and nothing it prints is financial or lab-record output.

> **📌 Status in one line:** The physics, optics, pricer, quiz/viva logic, PDF export, and reel daemon are implemented and were tested. The shipped readout weights are trained on **synthetic data**, so no performance claim about the organoid reservoir has been made yet.

---

## ⚡ Quickstart

```bash
npm install            # also copies the ONNX Runtime WASM to public/ort/
npm run typecheck
npm run dev            # http://localhost:5173  (COOP/COEP headers enabled)
npm run build
```

**Requirements:**
* A Chromium-based browser (WebGPU optional; the app falls back to a CPU Hodgkin-Huxley path)
* Node 18+
* Python 3 with `numpy` for training (`onnx`, `onnxruntime` optional, used to validate the exported model)

> 💡 **About the live demo:** A Google Sites embed cannot set the `COOP/COEP` headers that `SharedArrayBuffer` needs, and iframes may restrict WebGPU. Expect the app to report *snapshot mode* and possibly the CPU physics backend there. The badge at the top of the console shows which backend is active. For full behaviour, run it locally with `npm run dev`.

---

## 🔬 How It Works

1. ⚙️ **Physics Worker** (`src/core-physics/workers/physics.worker.ts`, `wgsl/organoid_rk4.wgsl`): 10,240 neurons with Hodgkin-Huxley dynamics, a 4-state ChR2 photocycle, Michaelis-Menten ATP/glucose kinetics, Q10 temperature scaling, and bio-heat relaxation. The WebGPU compute shader uses RK4; a CPU path implements the same model with a different integrator. Telemetry is shared with the UI through a `SharedArrayBuffer` (or posted snapshots when unavailable).
2. 🔦 **Optics** (`src/optics/gerchberg_saxton.ts`): Real 32x32 FFT-based Gerchberg-Saxton phase retrieval. The achieved focal-plane intensity becomes the per-region two-photon drive gain. Excitation uses `P² · exp(-2·depth/l_s)` with axial confinement around the focal plane. The UI shows GS fidelity and spot uniformity.
3. 🧠 **Reservoir & Readout** (`src/data-reservoir/heston_lsm.ts`): Neurons are pooled into a 4x4x4 spatial grid (64 bins). Each bin passes through a leaky integrator with its own time constant (0.2–5 s). A ridge readout (ONNX Runtime Web, or the identical JS product) outputs `[annualised realised vol, vol-of-vol proxy, signed 10 s forward return in bps]`.
4. 📈 **Heston-Inspired Variance Filter & Pricer**: Variance is filtered toward the readout's prediction, with vol-of-vol clamped to satisfy the Feller condition. A real Heston characteristic-function pricer (`heston_pricer.ts`, Gil-Pelaez inversion) prices a 30-day ATM call.
5. 📊 **Market Feed**: Binance public `btcusdt@depth20@100ms` (read-only). Top-10 book imbalance and a composite book-pressure signal drive the SLM target. If the feed is down, a clearly labelled synthetic random walk is used.
6. 💸 **Ledger (Toy)**: Costs and edge are both in bps with explicit conversion constants in `LEDGER`. It throttles the laser on photothermal limit (>38.5 °C), ATP depletion (<1 mM), or cost > edge. The "UPI" string is a placeholder demo URI, not a QR code and not a settlement.

---

## 🗂️ Node Inventory (Blueprint Numbering)

| Node | Subsystem | Status | Notes |
| :--- | :--- | :--- | :--- |
| **01** | Dossier Engine | 🟢 Implemented | Donor ID is never written to the PDF (true redaction); SHA-256 of the telemetry record is printed |
| **02** | Image Calibrator | 🔴 **Not implemented** | No micrograph cropping or scale bars |
| **03** | Manifest Builder | 🟡 Partial | One-page A4 telemetry summary; not a full SOP compiler |
| **04** | Sandbox Examiner | 🟢 Implemented | 90 s timer that locks the quiz; shuffled options; parameterised Feller and attenuation questions |
| **05** | Procedural ELN | 🟡 Partial | Seeded baseline jitter on typed text (not handwriting); watermarked `SIMULATED DATA` |
| **06** | Sentinel Daemon | 🟡 Partial | In-browser thermal/ATP alert and auto-throttle; Android background service not built |
| **07** | Micro-Ledger | 🟢 Implemented (toy) | Unit-consistent bps model; demo URI string only |
| **08** | Interactive Canvas | 🟢 Implemented | WebGL2 point cloud plus dendritic filaments; WebGPU is used for physics, not rendering. Frame rate not benchmarked |
| **09** | Focus Gatekeeper | 🟢 Implemented | Randomised, shuffled optics/Heston questions; the unlock timestamp is editable, so it is self-discipline, not security |
| **10** | Board Scraper | 🟢 Implemented | Queries the public Grants.gov API; seed rows are labelled DEMO |
| **11** | Bug Sniffer | 🟡 Partial | Logs click coordinates; does not generate prompts or apply corrections |
| **12** | Markdown Logger | 🔴 **Not implemented** | |
| **13** | Palette Extractor | 🟡 Partial | Four fixed GLSL palettes (Viridis, GCaMP6f, mCherry, DAPI); no CSS extraction |
| **14** | RAM Conservator | 🟢 Implemented | Discards idle tabs, skipping active, pinned, audible and app tabs; not linked to worker heap size |
| **15** | Video Abstracter | 🟡 Partial | Directory-watching FFmpeg daemon (9:16, 1080x1920); no in-app recorder (an `OffscreenCanvas` has no `captureStream`); bursts come from an optional `.events.json` |
| **16** | Adversarial Defense Examiner | 🟢 Implemented | Rule-based claim checker with keyword-stuffing detection; not an LLM |
| **17** | UI Refactor | 🟢 Implemented | The dark-mode console (`index.html`, `src/main.ts`) |

---

## 🎯 Training the Readout on Real Data

The shipped `public/models/*` are a **synthetic smoke test** (the UI labels them as such). To train on real reservoir states:

1. Run the app with the live L2 feed connected, click **Record reservoir states + L2 features**, let it run 30+ minutes, then stop to download `reservoir_recording.csv`.
2. Run the trainer:
   ```bash
   npm run train:recorded -- reservoir_recording.csv
   ```

The trainer standardises targets, picks a ridge lambda per output on a validation block, uses a time-ordered split with a purge gap, and reports **held-out per-column R²** next to four baselines (zero-skill, EWMA realised vol, linear on raw L2 features, random-feature reservoir). If the organoid reservoir does not beat the baselines, that is the result. On the synthetic set only the volatility output is learnable; vol-of-vol and return R² are about 0, as expected for a random-walk price.

---

## 🧩 Chrome Extension & Spike-Reel Daemon

### Chrome Extension (Nodes 9, 10, 14)
Navigate to `chrome://extensions` ➔ enable **Developer mode** ➔ click **Load unpacked** ➔ select `extensions/manifest-v3/`.

### Spike-Reel Daemon (Node 15)
```bash
python extensions/python-daemon/spike_reel_daemon.py --watch ~/Downloads   # process each new .webm once
python extensions/python-daemon/spike_reel_daemon.py raw.webm              # one-shot
```
Requires `ffmpeg` on `PATH`. If `raw.events.json` (`{"bursts":[{"t":12.3,"dur":6}]}`) sits next to the video, each burst becomes its own reel.

---

## ✅ What Was Verified

| Check | Result |
| :--- | :--- |
| `tsc --noEmit`, `vite build` | 🟢 Pass |
| `organoid_rk4.wgsl` compiled and run in Dawn on a software Vulkan GPU | 🟢 0 compile messages, no NaNs; ATP and temperature match the CPU model, firing rates within ~2–4% |
| Heston pricer vs independent Monte Carlo (4 parameter sets) | 🟢 Within 0.2%; matches Black-Scholes at σ = 0.01 |
| Gerchberg-Saxton | 🟢 Fidelity 0.935 (1 iteration) to 0.998 (30 iterations) |
| Trainer and ONNX export | 🟢 Runs; `onnx.checker` and onnxruntime parity ≈ 5e-7 |
| Quiz answer positions, viva scorer, PDF redaction and render, FFmpeg daemon | 🧪 Tested |

> ⚠️ **Not verified:** Behaviour on a real GPU and browser, WebGL rendering, the live Binance socket, Grants.gov responses, the Chrome extension at runtime, and the frame rate.

---

## ⚠️ Known Limitations

* **Dual Implementations:** The CPU path and the WGSL shader are separate implementations of the same model; keep them in sync when editing either.
* **Synthetic Weights:** Readout weights are synthetic-trained until you record and retrain. Reservoir features at runtime come from simulated neurons, so a model trained on other data will not transfer.
* **Optical Attenuation:** The two-photon model uses a ballistic attenuation law `exp(-2z/l_s)`. The Henyey-Greenstein angular phase function is not simulated.
* **Fixed Heston Parameters:** Heston parameters are fixed (κ = 2.0, θ = 0.20, ρ = -0.65); only σ and v₀ vary at runtime.
* **Wall-Clock Firing Rate:** The in-app firing rate is a wall-clock rate (spikes per neuron per real second), not biological Hz.

---

## 📂 Project Layout

~~~text
├── index.html                           # Console UI
├── src/
│   ├── main.ts                          # Orchestrator, camera, controls, recorder
│   ├── constants.ts                     # Shared constants & telemetry layout
│   ├── core-physics/workers/            # Physics + WebGL2 worker
│   ├── core-physics/wgsl/               # RK4 compute shader
│   ├── optics/gerchberg_saxton.ts       # FFT-based GS
│   ├── data-reservoir/                  # LSM readout, Heston filter & pricer
│   └── nodes/                           # Quiz/viva engine, PDF suite
├── extensions/
│   ├── manifest-v3/                     # Chrome extension
│   └── python-daemon/                   # FFmpeg reel daemon
├── data-reservoir/training/             # Ridge trainer and ONNX exporter
└── public/models/                       # Readout weights (.json) and graph (.onnx)
~~~

## 📸 Screenshots

<img width="1696" height="914" alt="Screenshot 2026-11-06 at 5 16 53 PM" src="https://github.com/user-attachments/assets/798b8c1e-5a58-45cf-b821-2d613afa9ec6" />
<img width="1695" height="402" alt="Screenshot 2026-11-06 at 5 17 32 PM" src="https://github.com/user-attachments/assets/4bfd1184-0c83-4832-8bd6-8b10ffa8f15b" />


---

## 🛑 Disclaimer

**Simulation only.** Not financial advice, not a validated biophysical model, and PDFs it produces are watermarked as simulated data and must not be used as laboratory records.
