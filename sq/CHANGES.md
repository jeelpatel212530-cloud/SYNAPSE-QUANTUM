# SYNAPSE-QUANTUM: fixes applied

Drop these files over your repo (same layout as your README). **Keep your own `src/core-physics/wgsl/organoid_rk4.wgsl`:
it was not in the upload, so it is untouched and unreviewed.** Then `npm install` (copies the ORT WASM to `public/ort/`), `npm run dev`.

## Bugs found while fixing (not in my first review)
| File | Bug | Effect |
|---|---|---|
| physics.worker.ts | `floatprojScale` typo in the soma vertex shader | Shader never compiled, so WebGL2 init returned false and the app silently used a 2D fallback that **never drew anything** |
| physics.worker.ts | `GPUMapMode.Read` (should be `READ`) | `mapAsync(undefined)` throws: the **WebGPU readback path could never work** |
| physics.worker.ts | One thrown frame ended the loop | Added try/catch + GPU-to-CPU fallback; loop always reschedules |
| physics.worker.ts | CPU fallback never updated m/h/n, no ChR2 cycle, temp capped at 37.06 C | No real spiking; sentinel (38.5 C) could never fire |
| main.ts | Fixed 760x450 canvas stretched by CSS | Distorted aspect ratio; now sized to device pixels + ResizeObserver |
| main.ts | Firing rate = instantaneous flag sampled at 10 Hz x 60 | Meaningless; worker now counts per-frame spikes |
| popup.js | Scraped titles written with `innerHTML` | XSS from network data; now `textContent` |
| main.ts / heston_lsm.ts | Live feed's `spreadVelocity` was computed but never sent to the worker | Slider was the only input; now "Drive from live L2" checkbox |

## Everything from my review, fixed
1. **Variance drift**: readout now predicts realised vol (annualised); variance is filtered toward it (full truncation, no runaway to the clamp).
2. **Metrics**: standardised targets, per-output lambda, time-ordered split with purge gap, **per-column held-out R2**, JSON stores the ablation.
3. **Quiz answers**: shuffled; Feller and attenuation questions are parameterised with computed keys (app + extension).
4. **Viva scorer**: claim-based (formula/relationship checks), keyword stuffing capped. `"scattering henyey ... "` now scores 0 (was 100).
5. **Worker import on main thread**: new `src/constants.ts`.
6. **SharedArrayBuffer fallback**: checks `crossOriginIsolated`; otherwise worker posts snapshots and the badge says so.
7. **Redaction / hashing**: donor ID is never written; SHA-256 of the record printed in the PDF; `SIMULATED DATA` watermark; jitter seeded from the hash.
8. **Reservoir design**: 4x4x4 spatial bins (not `i % 64`), per-bin leaky integrators (0.2-5 s), heterogeneous drive from the real GS hologram and tonic current.
9. **Training on real states**: app has a "Record" button (live feed only) that writes `reservoir_recording.csv`; trainer takes `--csv`.
10. **Ablation**: organoid reservoir vs zero-skill, EWMA vol, linear on raw L2, random-feature reservoir, on the same split.
11. **Heston**: real characteristic-function pricer (`heston_pricer.ts`) replaces the 0.3989 hack; blueprint notation issues are doc-only, left for you.
12. **Gerchberg-Saxton**: real 32x32 FFT-based GS (`optics/gerchberg_saxton.ts`); its achieved focal intensity drives per-region gain; fidelity/uniformity shown in the UI.
13. **Ledger**: all in bps with explicit toy constants (`LEDGER`), per-window not cumulative, hysteresis, plus hard throttles for temp > 38.5 C and ATP < 1 mM. Alpha keeps its sign.
14. **Feed**: `btcusdt@depth20@100ms` (top-10 imbalance); testnet URL in a comment.
15. **Feller**: sigma clamped to 0.99 sqrt(2 kappa theta). Note: I changed theta 0.045 to 0.20 (BTC-like variance), kappa 2.0.
16. **Node 10**: queries the public Grants.gov API; seed rows labelled DEMO; no invented stipends.
17. **Node 5 / UPI**: SIMULATED watermark, seeded jitter; UPI URI uses a placeholder VPA and says DEMO.
18. **Node 14**: skips pinned/audible/active/app tabs; `activeTab` dropped.
19. **Node 15**: real directory watcher, waits for stable file size, 9:16 crop + scale to 1080x1920, per-burst reels from an optional `.events.json`.
20. **ONNX**: kept the hand-rolled writer (it validates), now checked with `onnx.checker` + onnxruntime parity in the trainer. ORT WASM self-hosted for COEP.

## What I actually ran
- `tsc --noEmit`: passes (with a stub WGSL).
- Heston pricer vs independent Monte Carlo on 4 parameter sets: within 0.2%; matches Black-Scholes at sigma = 0.01.
- GS: fidelity 0.935 (1 iter) to 0.998 (30 iters).
- CPU physics (Node): rest = 0 Hz and ATP 3.15 mM; 35 mW centre = firing + ATP drawdown; heating crosses 38.5 C near 80 mW (39.2 C at 120 mW).
- Trainer + ONNX export: runs; ONNX parity vs numpy 5e-7.
- Viva, quiz answer-position distribution (uniform), PDF render + text extraction, FFmpeg daemon end to end (1080x1920 H.264).

## NOT verified (needs your machine)
- **WebGPU / WGSL path**: the WGSL file wasn't provided and I can't run a browser GPU. The CPU model and the WGSL are separate implementations: mirror depth/axial/heating/ATP logic into the shader or send it to me.
- WebGL2 rendering (typo fixed by inspection), the Binance socket, Grants.gov (response parsing is defensive but written from memory of the API), the Chrome extension at runtime, `vite build`.

## Read this before trusting any number
- **`public/models/*` ship trained on SYNTHETIC data** (the UI labels this). They only prove the pipeline. Record 30+ min of live feed, then `npm run train:recorded -- reservoir_recording.csv`. On the synthetic set only the vol column is learnable; vol-of-vol and return R2 are ~0, as they should be for a random-walk price.
- Your README says R2 = 0.8749; the shipped JSON said 0.6921 (pooled, in-sample). Neither is a result.
- Still missing: Node 2 scale bars, Node 6 Android sentinel, Node 12 derivation logger, and the MediaRecorder-to-daemon handoff (an `OffscreenCanvas` has no `captureStream`; it needs worker-side WebCodecs or a screen-capture route).
