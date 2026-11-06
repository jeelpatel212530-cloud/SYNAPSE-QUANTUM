#!/usr/bin/env python3
"""
SYNAPSE-QUANTUM: Liquid State Machine ridge readout trainer (fixed).

Usage
  python train_heston_readout_onnx.py                      # synthetic SMOKE TEST (not evidence)
  python train_heston_readout_onnx.py --csv rec.csv        # train on data recorded from the app

What changed vs. the first version
  * Targets are STANDARDISED and each output gets its own ridge lambda (chosen on a validation
    block), so the small-scale columns are no longer crushed by the alpha column.
  * Time-ordered train / val / test split with a purge gap (no look-ahead leakage).
  * Metrics are PER COLUMN and on HELD-OUT data (the old R^2 was pooled and in-sample).
  * Baselines are reported next to the reservoir: zero-skill mean, EWMA realised vol, linear
    regression on raw L2 features, and a random-feature reservoir. If the organoid reservoir
    does not beat them, that is the (legitimate) result.
  * Standardisation is folded into W, b so the browser runs a plain x @ W + b.
  * Exported ONNX is validated (onnx.checker + onnxruntime parity) when those packages exist.

Targets (all computed from the NEXT 30-60 s of mid-price, never from the features)
  col 0  realised vol, annualised (decimal)           -> Heston variance filter input
  col 1  vol-of-vol proxy: CoV of 5 s realised vol    -> sigma (clamped to Feller in the app)
  col 2  signed forward return over 10 s, in bps      -> alpha signal
CSV columns: t_ms, s0..s63, mid, spread_bps, imbalance   (10 Hz, as written by the app recorder)
"""
import argparse
import json
from pathlib import Path

import numpy as np

BINS = 64
OUT = 3
SEC_PER_YEAR = 365 * 24 * 3600
H_VOL, H_VOV, H_RET = 30, 60, 10   # horizons in seconds (1 Hz samples)
PURGE = H_VOV                       # samples dropped between splits


# ----------------------------------------------------------------------------- ONNX writer
def encode_varint(value: int) -> bytes:
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value & 0x7F)
    return bytes(out)


def pb_field(num: int, wire: int, payload: bytes) -> bytes:
    tag = (num << 3) | wire
    if wire == 0:
        return encode_varint(tag) + payload
    if wire == 2:
        return encode_varint(tag) + encode_varint(len(payload)) + payload
    raise ValueError(wire)


def pb_string(num, text): return pb_field(num, 2, text.encode())
def pb_varint(num, val): return pb_field(num, 0, encode_varint(val))


def tensor_proto(name, dims, data):
    msg = bytearray()
    for d in dims:
        msg += pb_varint(1, d)
    msg += pb_varint(2, 1) + pb_string(8, name)
    msg += pb_field(9, 2, np.asarray(data, dtype="<f4").tobytes())
    return bytes(msg)


def value_info(name, dims):
    shape = bytearray()
    for d in dims:
        shape += pb_field(1, 2, pb_varint(1, d))
    tt = pb_varint(1, 1) + pb_field(2, 2, bytes(shape))
    return pb_string(1, name) + pb_field(2, 2, pb_field(1, 2, tt))


def node_proto(op, inputs, outputs, name):
    msg = bytearray()
    for i in inputs: msg += pb_string(1, i)
    for o in outputs: msg += pb_string(2, o)
    return bytes(msg + pb_string(3, name) + pb_string(4, op))


def serialize_linear_onnx(W: np.ndarray, b: np.ndarray) -> bytes:
    n_in, n_out = W.shape
    graph = (
        pb_field(1, 2, node_proto("MatMul", ["reservoir_state", "W_ridge"], ["matmul_out"], "RidgeMatMul"))
        + pb_field(1, 2, node_proto("Add", ["matmul_out", "b_ridge"], ["heston_out"], "RidgeAdd"))
        + pb_string(2, "SynapseQuantumHestonLSMReadout")
        + pb_field(5, 2, tensor_proto("W_ridge", [n_in, n_out], W))
        + pb_field(5, 2, tensor_proto("b_ridge", [1, n_out], b.reshape(1, n_out)))
        + pb_field(11, 2, value_info("reservoir_state", [1, n_in]))
        + pb_field(12, 2, value_info("heston_out", [1, n_out]))
    )
    opset = pb_string(1, "") + pb_varint(2, 13)
    return pb_varint(1, 7) + pb_string(2, "synapse-quantum-ridge-exporter") + pb_string(3, "2.0.0") + pb_field(7, 2, graph) + pb_field(8, 2, opset)


# ----------------------------------------------------------------------------- data
def sim_heston_prices(n_sec: int, rng):
    """Synthetic 1 Hz mid-price with a STATIONARY CIR-like variance (10 min mean reversion)."""
    dt = 1.0 / SEC_PER_YEAR
    theta, tau_rev, std_v, rho = 0.2, 600.0, 0.08, -0.5
    noise = std_v * np.sqrt(2.0 / tau_rev)
    v = np.empty(n_sec); v[0] = theta
    r = np.zeros(n_sec)
    for i in range(1, n_sec):
        z1 = rng.standard_normal(); z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal()
        vp = max(v[i - 1], 1e-3)
        v[i] = max(vp + (theta - vp) / tau_rev + noise * np.sqrt(vp / theta) * z2, 1e-3)
        r[i] = np.sqrt(vp * dt) * z1
    return 64000 * np.exp(np.cumsum(r)), v


def synthetic_recording(n_sec=20000, seed=42):
    """SMOKE-TEST data only: L2 features correlated with true variance + a 64-bin leaky tanh reservoir."""
    rng = np.random.default_rng(seed)
    mid, v = sim_heston_prices(n_sec, rng)
    vol = np.sqrt(v)
    imb = np.tanh(0.8 * rng.standard_normal(n_sec) + 0.4 * np.sign(np.r_[0, np.diff(np.log(mid))]))
    spread = 0.01 + 0.02 * (vol / 0.45) ** 2 + 0.004 * np.abs(rng.standard_normal(n_sec))
    feats = np.column_stack([spread, imb, np.abs(np.r_[0, np.diff(np.log(mid))]) * 1e4])
    proj = rng.standard_normal((3, BINS)) * 1.2
    taus = 0.2 * 25 ** (np.arange(BINS) / (BINS - 1))
    S = np.zeros((n_sec, BINS)); s = np.zeros(BINS)
    for i in range(n_sec):
        x = np.tanh((feats[i] - feats.mean(0)) / (feats.std(0) + 1e-9) @ proj + 0.15 * rng.standard_normal(BINS))
        s += (1 - np.exp(-1.0 / taus)) * (x - s)
        S[i] = s
    return S.astype(np.float32), mid, spread, imb


def load_csv(path):
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    S = raw[:, 1:1 + BINS][::10]                      # 10 Hz -> 1 Hz
    mid = raw[:, 1 + BINS][::10]
    spread = raw[:, 2 + BINS][::10]
    imb = raw[:, 3 + BINS][::10]
    return S.astype(np.float32), mid, spread, imb


def build_targets(mid):
    """Targets from FUTURE prices only. Returns Y (n, 3) and valid mask."""
    n = len(mid)
    lr = np.r_[0.0, np.diff(np.log(mid))]
    Y = np.full((n, OUT), np.nan)
    c2 = np.cumsum(lr ** 2)
    for i in range(n - H_VOV - 1):
        w = lr[i + 1:i + 1 + H_VOL]
        Y[i, 0] = np.sqrt(np.mean(w ** 2) * SEC_PER_YEAR)
        blk = lr[i + 1:i + 1 + H_VOV]
        rv5 = np.sqrt(np.mean(blk[: (H_VOV // 5) * 5].reshape(-1, 5) ** 2, axis=1) * SEC_PER_YEAR)
        Y[i, 1] = rv5.std() / (rv5.mean() + 1e-12)
        Y[i, 2] = (np.log(mid[i + H_RET]) - np.log(mid[i])) * 1e4
    return Y, ~np.isnan(Y).any(axis=1)


# ----------------------------------------------------------------------------- models
def ridge_fit(Xz, Yz, lam):
    return np.linalg.solve(Xz.T @ Xz + lam * np.eye(Xz.shape[1]), Xz.T @ Yz)


def per_col_r2(Y, P):
    ss_res = ((Y - P) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    return 1 - ss_res / np.maximum(ss_tot, 1e-12)


def fit_ridge_per_output(Xtr, Ytr, Xva, Yva, grid=(0.1, 1, 10, 100, 1000, 1e4)):
    """Standardise, choose lambda per output on validation, return folded (W, b) and chosen lambdas."""
    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-8
    ym, ys = Ytr.mean(0), Ytr.std(0) + 1e-12
    Xz, Yz = (Xtr - xm) / xs, (Ytr - ym) / ys
    Xvz = (Xva - xm) / xs
    Wz = np.zeros((Xtr.shape[1], OUT)); lams = []
    for j in range(OUT):
        best = None
        for lam in grid:
            w = ridge_fit(Xz, Yz[:, j:j + 1], lam)[:, 0]
            err = np.mean(((Xvz @ w) * ys[j] + ym[j] - Yva[:, j]) ** 2)
            if best is None or err < best[0]:
                best = (err, lam, w)
        Wz[:, j] = best[2]; lams.append(best[1])
    W = (Wz / xs[:, None]) * ys[None, :]                     # fold standardisation
    b = ym - (xm / xs) @ Wz * ys
    return W.astype(np.float32), b.astype(np.float32), lams


def ewma_vol_baseline(mid, span=30):
    lr = np.r_[0.0, np.diff(np.log(mid))]
    a = 2 / (span + 1); v = np.zeros_like(lr); v[0] = lr[0] ** 2
    for i in range(1, len(lr)):
        v[i] = (1 - a) * v[i - 1] + a * lr[i] ** 2
    return np.sqrt(v * SEC_PER_YEAR)


def evaluate_all(S, mid, spread, imb, seed=0):
    Y, ok = build_targets(mid)
    n = len(mid)
    raw = np.column_stack([spread, imb, np.abs(imb), np.r_[0, np.abs(np.diff(np.log(mid)))] * 1e4])
    idx = np.where(ok)[0]
    n_tr, n_va = int(0.6 * n), int(0.2 * n)
    tr = idx[idx < n_tr - PURGE]
    va = idx[(idx >= n_tr) & (idx < n_tr + n_va - PURGE)]
    te = idx[idx >= n_tr + n_va]
    if min(len(tr), len(va), len(te)) < 100:
        raise SystemExit("Not enough data after splitting; record a longer session (>= ~30 min).")

    results, models = {}, {}

    def score(name, Ptest):
        results[name] = {"r2": [float(x) for x in per_col_r2(Y[te], Ptest)],
                         "rmse": [float(x) for x in np.sqrt(((Y[te] - Ptest) ** 2).mean(0))]}

    score("zero_skill_mean", np.tile(Y[tr].mean(0), (len(te), 1)))
    ew = ewma_vol_baseline(mid)
    P = np.tile(Y[tr].mean(0), (len(te), 1)); P[:, 0] = ew[te]
    score("ewma_realised_vol", P)
    W, b, lam = fit_ridge_per_output(raw[tr], Y[tr], raw[va], Y[va]); score("linear_on_raw_L2", raw[te] @ W + b)
    rng = np.random.default_rng(seed)
    Rproj = rng.standard_normal((raw.shape[1], BINS)); rz = (raw - raw[tr].mean(0)) / (raw[tr].std(0) + 1e-8)
    RF = np.tanh(rz @ Rproj)
    W, b, lam = fit_ridge_per_output(RF[tr], Y[tr], RF[va], Y[va]); score("random_feature_reservoir", RF[te] @ W + b)
    W, b, lam = fit_ridge_per_output(S[tr], Y[tr], S[va], Y[va]); score("organoid_reservoir", S[te] @ W + b)
    models["W"], models["b"], models["lam"] = W, b, lam
    # refit final model on train+val
    trv = np.r_[tr, va]
    Wf, bf, lamf = fit_ridge_per_output(S[trv], Y[trv], S[va], Y[va])
    return results, Wf, bf, lamf, (len(tr), len(va), len(te))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="recording produced by the web app")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if a.csv:
        S, mid, spread, imb = load_csv(a.csv); source = Path(a.csv).name
    else:
        S, mid, spread, imb = synthetic_recording(seed=a.seed); source = "SYNTHETIC smoke test"
        print("!! No --csv given: training on synthetic data. This validates the pipeline, NOT the organoid.")

    results, W, b, lam, sizes = evaluate_all(S, mid, spread, imb, seed=a.seed)
    print(f"split sizes (train/val/test): {sizes}   lambdas per output: {lam}")
    print(f"{'model':28s} {'R2 vol':>8s} {'R2 vov':>8s} {'R2 ret':>8s}")
    for k, v in results.items():
        print(f"{k:28s} " + " ".join(f"{x:8.3f}" for x in v["r2"]))

    root = Path(__file__).resolve().parents[2]
    models_dir = root / "public" / "models"; models_dir.mkdir(parents=True, exist_ok=True)
    onnx_bytes = serialize_linear_onnx(W, b)
    (models_dir / "heston_readout.onnx").write_bytes(onnx_bytes)
    payload = {
        "model": "SYNAPSE-QUANTUM Ridge Readout (64x3)", "trained_on": source,
        "input_dim": BINS, "output_dim": OUT, "lambdas": lam,
        "outputs": ["realised_vol_annualised", "vol_of_vol_proxy_CoV", "forward_ret_10s_bps"],
        "metrics": {"test_r2": results["organoid_reservoir"]["r2"], "test_rmse": results["organoid_reservoir"]["rmse"]},
        "ablation": results,
        "weights": W.tolist(), "bias": b.tolist(),
    }
    (models_dir / "heston_readout_weights.json").write_text(json.dumps(payload, indent=2))

    # validate the ONNX we just wrote
    try:
        import onnx, onnxruntime as ort
        onnx.checker.check_model(onnx.load_from_string(onnx_bytes))
        sess = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
        x = np.random.default_rng(0).standard_normal((1, BINS)).astype(np.float32)
        diff = float(np.abs(sess.run(None, {"reservoir_state": x})[0] - (x @ W + b)).max())
        print(f"ONNX validated (checker + onnxruntime parity, max |diff| = {diff:.2e})")
    except ImportError:
        print("onnx/onnxruntime not installed: skipped ONNX validation (pip install onnx onnxruntime)")
    print(f"saved -> {models_dir}")


if __name__ == "__main__":
    main()
