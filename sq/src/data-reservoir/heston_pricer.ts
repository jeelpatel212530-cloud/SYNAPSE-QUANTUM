// ============================================================================
// Heston (1993) European call via the characteristic function (Gil-Pelaez inversion),
// using the Albrecher et al. "little trap" formulation for numerical stability.
// Pure TypeScript, no DOM: unit-testable in Node.
// ============================================================================

type C = { re: number; im: number };
const c = (re: number, im = 0): C => ({ re, im });
const add = (a: C, b: C): C => c(a.re + b.re, a.im + b.im);
const sub = (a: C, b: C): C => c(a.re - b.re, a.im - b.im);
const mul = (a: C, b: C): C => c(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
const scale = (a: C, k: number): C => c(a.re * k, a.im * k);
const div = (a: C, b: C): C => {
  const d = b.re * b.re + b.im * b.im;
  return c((a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d);
};
const exp = (a: C): C => { const e = Math.exp(a.re); return c(e * Math.cos(a.im), e * Math.sin(a.im)); };
const log = (a: C): C => c(Math.log(Math.hypot(a.re, a.im)), Math.atan2(a.im, a.re));
const sqrt = (a: C): C => {
  const r = Math.hypot(a.re, a.im);
  const re = Math.sqrt((r + a.re) / 2);
  const im = Math.sign(a.im || 1) * Math.sqrt(Math.max(0, (r - a.re) / 2));
  return c(re, im);
};

export interface HestonParams {
  S: number; K: number; tau: number; r: number;
  v0: number; kappa: number; theta: number; sigma: number; rho: number;
}

/** Characteristic function f_j(u) of ln S_T, j = 1 or 2. */
function charFn(p: HestonParams, u: number, j: 1 | 2): C {
  const { S, tau, r, v0, kappa, theta, sigma, rho } = p;
  const uj = j === 1 ? 0.5 : -0.5;
  const b = j === 1 ? kappa - rho * sigma : kappa;
  const iu = c(0, u);
  const rsiu = scale(iu, rho * sigma);
  const bm = sub(c(b), rsiu);                                    // b - rho*sigma*i*u
  // d = sqrt( (rho*sigma*i*u - b)^2 - sigma^2 (2 u_j i u - u^2) )
  const t1 = mul(sub(rsiu, c(b)), sub(rsiu, c(b)));
  const t2 = scale(sub(scale(iu, 2 * uj), c(u * u)), sigma * sigma);
  const d = sqrt(sub(t1, t2));
  const g = div(sub(bm, d), add(bm, d));
  const edt = exp(scale(d, -tau));
  const oneMinusGedt = sub(c(1), mul(g, edt));
  const oneMinusG = sub(c(1), g);
  const C_ = add(
    scale(iu, r * tau),
    scale(sub(scale(sub(bm, d), tau), scale(log(div(oneMinusGedt, oneMinusG)), 2)), (kappa * theta) / (sigma * sigma)),
  );
  const D_ = scale(mul(div(sub(bm, d), c(1)), div(sub(c(1), edt), oneMinusGedt)), 1 / (sigma * sigma));
  return exp(add(add(C_, scale(D_, v0)), scale(iu, Math.log(S))));
}

export function hestonCallPrice(p: HestonParams, uMax = 200, steps = 800): number {
  const lnK = Math.log(p.K);
  const h = uMax / steps;
  let I1 = 0, I2 = 0;
  for (let k = 0; k < steps; k++) {
    const u = (k + 0.5) * h; // midpoint rule avoids the u = 0 singularity
    const e = exp(c(0, -u * lnK));
    const iu = c(0, u);
    I1 += div(mul(e, charFn(p, u, 1)), iu).re * h;
    I2 += div(mul(e, charFn(p, u, 2)), iu).re * h;
  }
  const P1 = 0.5 + I1 / Math.PI;
  const P2 = 0.5 + I2 / Math.PI;
  return p.S * P1 - p.K * Math.exp(-p.r * p.tau) * P2;
}

/** Black-Scholes call, used as the reference in tests and as a sanity bound. */
export function bsCall(S: number, K: number, tau: number, r: number, vol: number): number {
  const d1 = (Math.log(S / K) + (r + 0.5 * vol * vol) * tau) / (vol * Math.sqrt(tau));
  const d2 = d1 - vol * Math.sqrt(tau);
  const N = (x: number) => 0.5 * (1 + erf(x / Math.SQRT2));
  return S * N(d1) - K * Math.exp(-r * tau) * N(d2);
}
function erf(x: number): number {
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return x >= 0 ? y : -y;
}
