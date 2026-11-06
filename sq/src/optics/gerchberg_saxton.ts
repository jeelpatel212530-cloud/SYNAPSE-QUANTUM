// ============================================================================
// Real Gerchberg-Saxton phase retrieval on a 32x32 SLM grid (radix-2 FFT).
// The worker uses the *achieved focal-plane intensity* of the retrieved hologram
// as the per-region two-photon drive gain, so the optics claims are testable.
// ============================================================================
import { SLM_GRID } from '../constants';
import { mulberry32 } from '../util/rng';

function fft1d(re: Float64Array, im: Float64Array, inverse: boolean): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i]; re[i] = re[j]; re[j] = tr;
      const ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((2 * Math.PI) / len) * (inverse ? 1 : -1);
    const wr = Math.cos(ang), wi = Math.sin(ang);
    const half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < half; k++) {
        const ur = re[i + k], ui = im[i + k];
        const xr = re[i + k + half], xi = im[i + k + half];
        const vr = xr * cr - xi * ci, vi = xr * ci + xi * cr;
        re[i + k] = ur + vr; im[i + k] = ui + vi;
        re[i + k + half] = ur - vr; im[i + k + half] = ui - vi;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

export function fft2d(re: Float64Array, im: Float64Array, N: number, inverse: boolean): void {
  const rr = new Float64Array(N), ri = new Float64Array(N);
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) { rr[x] = re[y * N + x]; ri[x] = im[y * N + x]; }
    fft1d(rr, ri, inverse);
    for (let x = 0; x < N; x++) { re[y * N + x] = rr[x]; im[y * N + x] = ri[x]; }
  }
  for (let x = 0; x < N; x++) {
    for (let y = 0; y < N; y++) { rr[y] = re[y * N + x]; ri[y] = im[y * N + x]; }
    fft1d(rr, ri, inverse);
    for (let y = 0; y < N; y++) { re[y * N + x] = rr[y]; im[y * N + x] = ri[y]; }
  }
}

// Fixed (seeded) hot-spot layout and feature mixing so targets are heterogeneous but reproducible.
const SPOTS = 12;
const spotRng = mulberry32(20260611);
const SPOT_POS: number[] = [];
const SPOT_GAIN: Array<[number, number, number]> = [];
for (let k = 0; k < SPOTS; k++) {
  SPOT_POS.push(Math.floor(spotRng() * SLM_GRID * SLM_GRID));
  SPOT_GAIN.push([(spotRng() - 0.5) * 3, (spotRng() - 0.5) * 3, (spotRng() - 0.5) * 2]);
}

/** Target focal-plane amplitude: each hot spot's weight is a different mix of the market features. */
export function buildTargetAmplitude(spreadVelocity: number, imbalance: number): Float64Array {
  const N = SLM_GRID;
  const target = new Float64Array(N * N).fill(0.25);
  for (let k = 0; k < SPOTS; k++) {
    const [a, b, c] = SPOT_GAIN[k];
    const w = 0.5 + 0.5 * Math.tanh(a * (spreadVelocity - 1.25) + b * imbalance + c);
    target[SPOT_POS[k]] = 0.5 + 0.5 * w;
  }
  return target;
}

export interface GSResult {
  phase: Float64Array;   // SLM phase mask, radians
  mask: Float32Array;    // achieved focal intensity, normalised to max = 1
  efficiency: number;    // overlap fidelity of achieved vs target amplitude (1 = perfect)
  uniformity: number;    // 1 - (max-min)/(max+min) over hot spots
}

export function gerchbergSaxton(target: Float64Array, iterations = 12, phase0?: Float64Array): GSResult {
  const N = SLM_GRID, n2 = N * N;
  const phase = new Float64Array(n2);
  if (phase0) phase.set(phase0);
  else { const r = mulberry32(7); for (let i = 0; i < n2; i++) phase[i] = r() * 2 * Math.PI; }

  const re = new Float64Array(n2), im = new Float64Array(n2);
  let tNorm = 0;
  for (let i = 0; i < n2; i++) tNorm += target[i] * target[i];
  tNorm = Math.sqrt(tNorm);

  for (let it = 0; it < iterations; it++) {
    for (let i = 0; i < n2; i++) { re[i] = Math.cos(phase[i]); im[i] = Math.sin(phase[i]); }
    fft2d(re, im, N, false);
    // keep focal-plane phase, impose target amplitude
    for (let i = 0; i < n2; i++) {
      const a = Math.hypot(re[i], im[i]) || 1e-12;
      const t = target[i] / tNorm * N; // unit-uniform input has |F| ~ N on average
      re[i] = (re[i] / a) * t; im[i] = (im[i] / a) * t;
    }
    fft2d(re, im, N, true);
    for (let i = 0; i < n2; i++) phase[i] = Math.atan2(im[i], re[i]); // uniform SLM illumination
  }

  for (let i = 0; i < n2; i++) { re[i] = Math.cos(phase[i]); im[i] = Math.sin(phase[i]); }
  fft2d(re, im, N, false);
  const inten = new Float64Array(n2);
  let total = 0, max = 0;
  for (let i = 0; i < n2; i++) { inten[i] = re[i] * re[i] + im[i] * im[i]; total += inten[i]; if (inten[i] > max) max = inten[i]; }

  let smin = Infinity, smax = 0, overlap = 0, tPow = 0;
  for (let i = 0; i < n2; i++) { overlap += Math.sqrt(inten[i]) * target[i]; tPow += target[i] * target[i]; }
  for (let k = 0; k < SPOTS; k++) {
    // compare hot spots after normalising by their requested amplitude^2
    const v = inten[SPOT_POS[k]] / (target[SPOT_POS[k]] * target[SPOT_POS[k]]);
    smin = Math.min(smin, v); smax = Math.max(smax, v);
  }
  const mask = new Float32Array(n2);
  for (let i = 0; i < n2; i++) mask[i] = (inten[i] / (max || 1)) as number;
  return {
    phase, mask,
    efficiency: (overlap * overlap) / ((total || 1) * (tPow || 1)),
    uniformity: smax + smin > 0 ? 1 - (smax - smin) / (smax + smin) : 0,
  };
}
