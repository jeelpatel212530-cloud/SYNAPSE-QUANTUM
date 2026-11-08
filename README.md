# 🧠 SYNAPSE-QUANTUM v3.0 | *Cleanroom Console*

**[🚀 Launch Live Demo](https://sites.google.com/view/synapse-quantum-v3-0-cleanroom/full-page-embed)** 

An offline-first, in-browser biophysics and quantitative finance simulation. SYNAPSE-QUANTUM models a **10,240-neuron cortical organoid** driven by a two-photon holographic SLM. Its firing activity is read out by a ridge-regression liquid state machine to feed a Heston-style volatility filter and pricer. 

> **Current Status:** The physics engine, optics, pricer, quiz/viva logic, PDF export, and reel daemon are fully implemented and tested. *Note: Shipped readout weights are trained on synthetic data; no performance claims regarding the organoid reservoir are made yet.*

---

## ⚡ Quickstart

Get the console running locally in seconds.

```bash
npm install            # Installs dependencies & copies ONNX Runtime WASM
npm run typecheck
npm run dev            # Launches on http://localhost:5173 (COOP/COEP enabled)
npm run build
Requirements:
Chromium-based browser (WebGPU optional; falls back to CPU Hodgkin-Huxley path)
Node 18+
Python 3 with numpy for training (onnx, onnxruntime optional for validation)
💡 Demo Note: The live Google Sites embed cannot set COOP/COEP headers required for SharedArrayBuffer, and iframes restrict WebGPU. The embed will run in snapshot mode using the CPU physics backend. For the full, unthrottled WebGPU experience, run it locally.

🔬 Architecture & How It Works
⚙️ Physics Engine (organoid_rk4.wgsl): 10,240 neurons modeled with Hodgkin-Huxley dynamics, a 4-state ChR2 photocycle, and Michaelis-Menten ATP/glucose kinetics. Features Q10 temperature scaling and bio-heat relaxation. Uses a WebGPU RK4 compute shader (with a CPU fallback).
🔦 Optics (gerchberg_saxton.ts): True 32x32 FFT-based Gerchberg-Saxton phase retrieval. Achieved focal-plane intensity dictates per-region two-photon drive gain, with realistic axial confinement.
🧠 Reservoir & Readout (heston_lsm.ts): Neurons pool into a 64-bin spatial grid. Bins pass through leaky integrators (0.2–5s time constants). An ONNX Web readout predicts annualized realized vol, vol-of-vol, and signed 10s forward return.
📈 Heston Pricer (heston_pricer.ts): A real Heston characteristic-function pricer (Gil-Pelaez inversion) prices a 30-day ATM call, filtering variance toward the readout's prediction while clamping vol-of-vol to satisfy the Feller condition.
📊 Market Feed: Ingests the public Binance btcusdt@depth20@100ms feed. Top-10 book imbalance and pressure signals drive the SLM target. (Defaults to a synthetic random walk if offline).
💸 Micro-Ledger: A toy economics model tracking costs and edge in bps. Throttles the laser based on photothermal limits (>38.5 °C), ATP depletion (<1 mM), or negative edge.
🗂️ Blueprint: Node Inventory
Node	Subsystem	Status	Notes
01	Dossier Engine	🟢 Implemented	Donor ID omitted (true redaction); SHA-256 of telemetry printed.
02	Image Calibrator	🔴 Unbuilt	No micrograph cropping or scale bars.
03	Manifest Builder	🟡 Partial	One-page A4 telemetry summary; not a full SOP compiler.
04	Sandbox Examiner	🟢 Implemented	90s locked timer, shuffled options, parameterized questions.
05	Procedural ELN	🟡 Partial	Seeded baseline jitter on typed text; watermarked SIMULATED DATA.
06	Sentinel Daemon	🟡 Partial	In-browser thermal/ATP alert & throttle (no Android service yet).
07	Micro-Ledger	🟢 Implemented	Toy unit-consistent bps model; demo URI string only.
08	Interactive Canvas	🟢 Implemented	WebGL2 point cloud + dendritic filaments. GPU strictly for physics.
09	Focus Gatekeeper	🟢 Implemented	Shuffled Heston/optics questions. Editable unlock timestamp.
10	Board Scraper	🟢 Implemented	Queries public Grants.gov API.
11	Bug Sniffer	🟡 Partial	Logs click coordinates; lacks LLM prompt generation.
12	Markdown Logger	🔴 Unbuilt	
13	Palette Extractor	🟡 Partial	Four fixed GLSL palettes (Viridis, GCaMP6f, mCherry, DAPI).
14	RAM Conservator	🟢 Implemented	Discards idle tabs (skips active, pinned, audible).
15	Video Abstracter	🟡 Partial	Directory-watching FFmpeg daemon for bursts; no in-app recorder.
16	Adversarial Def.	🟢 Implemented	Rule-based claim checker detecting keyword-stuffing.
17	UI Refactor	🟢 Implemented	Dark-mode console core (index.html, src/main.ts).
📈 Training on Real Data
The default public/models/* weights are a synthetic smoke test. To train the reservoir on actual market data:
Launch the app with the live L2 feed connected.
Click Record reservoir states + L2 features and capture 30+ minutes of data.
Stop recording, download reservoir_recording.csv, and run:
Bash
npm run train:recorded -- reservoir_recording.csv
The trainer standardizes targets, selects ridge lambdas per output, applies a time-ordered split with a purge gap, and outputs held-out per-column R² against four baselines (zero-skill, EWMA realized vol, linear raw L2, random-feature reservoir).
🧩 Extensions & Tools
Chrome Extension (Nodes 9, 10, 14)
Navigate to chrome://extensions → Developer mode → Load unpacked → Select extensions/manifest-v3/.
Spike-Reel Daemon (Node 15)
Processes bursts into individual reels (requires ffmpeg in your PATH):
Bash
python extensions/python-daemon/spike_reel_daemon.py --watch ~/Downloads   # Watch mode
python extensions/python-daemon/spike_reel_daemon.py raw.webm              # One-shot
✅ Verification Matrix
Subsystem Check	Result
tsc --noEmit, vite build	🟢 Pass
organoid_rk4.wgsl compile & Dawn execution	🟢 Pass (0 errors, no NaNs. Matches CPU ATP/temp, firing rates ±2-4%)
Heston pricer vs independent Monte Carlo	🟢 Pass (Matches BS at σ=0.01; within 0.2% on 4 parameter sets)
Gerchberg-Saxton Retrieval	🟢 Pass (Fidelity 0.935 at 1 iter ➔ 0.998 at 30 iters)
Trainer & ONNX export	🟢 Pass (onnx.checker & runtime parity ≈ 5e-7)
Quiz logic, PDF redaction, FFmpeg daemon	🧪 Tested
Live GPU, Browser rendering, Binance socket	⚠️ Unverified in current test suite
⚠️ Known Limitations
Dual Implementations: The CPU path and WGSL shader are distinct. You must manually keep them in sync if modifying the physics model.
Data Transferability: Reservoir features are derived from simulated neurons. Readout weights trained on other data will not transfer.
Optical Approximations: The two-photon model uses ballistic attenuation exp(-2z/l_s). The Henyey-Greenstein angular phase function is omitted.
Fixed Heston Parameters: κ = 2.0, θ = 0.20, ρ = -0.65. Only σ and v₀ are dynamic at runtime.
Time Scaling: The in-app firing rate is based on wall-clock time (spikes per neuron per real-world second), not biological Hz.
📂 Project Layout
Plaintext
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
📸 Console Previews
🛑 DISCLAIMER: This project is a research and teaching prototype. It is strictly a simulation. It does not constitute financial advice, nor is it a validated biophysical model. PDFs generated by the system are watermarked as simulated data and must never be used as valid laboratory records.
