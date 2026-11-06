// ============================================================================
// SYNAPSE-QUANTUM: Phase II Liquid State Machine readout + Heston-inspired variance filter.
//
// Honest description of what this does:
//   * Reservoir features: 64 spatial bins (4x4x4 grid) of the organoid, each passed through a
//     leaky integrator with its own time constant (fading memory) -> 64-dim state.
//   * Readout (ridge, ONNX or JS fallback) -> [realized vol (annualised), vol-of-vol proxy,
//     signed forward return in bps].
//   * Variance filter: v is pulled toward the predicted variance; sigma is clamped to satisfy the
//     Feller condition. European call price is computed with a *real* Heston pricer.
//   * L2 feed: Binance depth20@100ms (top-20 levels), mainnet public market data (read-only).
//   * Ledger: unit-consistent TOY model (everything in bps). Not financial advice or output.
// ============================================================================

import * as ort from 'onnxruntime-web';
import { RESERVOIR_BINS, F, spatialBin } from '../constants';
import { hestonCallPrice } from './heston_pricer';

// Testnet alternative: wss://stream.testnet.binance.vision/ws/btcusdt@depth20@100ms
const L2_URL = 'wss://stream.binance.com:9443/ws/btcusdt@depth20@100ms';
const BOOK_LEVELS = 10;
const SECONDS_PER_YEAR = 365 * 24 * 3600;

// Toy ledger conversion constants (documented assumptions, change freely)
export const LEDGER = { BPS_PER_JOULE: 100, BPS_PER_MM_ATP_DEFICIT: 2.0, ATP_REST_MM: 3.2 };
const DEMO_VPA = 'demo.vpa@upi'; // placeholder; replace with your own VPA if you really want a live URI

export interface HestonTelemetry {
  spotPrice: number;
  bidAskSpreadBps: number;
  spreadVelocity: number;     // composite book-pressure drive signal in [0.25, 3.5]
  imbalance: number;          // top-10 book imbalance in [-1, 1]
  feedLive: boolean;
  instantVariance: number;
  impliedVolatility: number;
  volOfVol: number;
  feller: boolean;
  hestonOptionFairValue: number;
  predictedAlphaBps: number;  // signed
  ledgerCostBps: number;
  throttled: boolean;
  throttleReason: string;
  upiSettlementUri: string;
  onnxStatus: string;
  l2FeedStatus: string;
  reservoirState: Float32Array;
  meanVm: number;
  meanAtp: number;
  maxTempC: number;
}

export class HestonLiquidStateEngine {
  private spotPrice = 64250.0;
  private bidAskSpreadBps = 1.4;
  private spreadVelocity = 1.25;
  private imbalance = 0;
  private feedLive = false;
  private variance = 0.2 * 0.2;
  private sigma = 0.4;
  private readonly kappa = 2.0;
  private readonly theta = 0.2; // long-run variance (vol ~ 45%, BTC-like)
  private readonly rho = -0.65;
  private readonly r = 0.05;

  private ridgeWeights = new Float32Array(RESERVOIR_BINS * 3);
  private ridgeBias = new Float32Array([0.45, 0.4, 0.0]);
  private onnxSession: ort.InferenceSession | null = null;
  private onnxStatus = 'LOADING READOUT...';
  private l2FeedStatus = 'CONNECTING...';
  private lastAsyncOutput: Float32Array | null = null;
  private isInferring = false;

  // reservoir features
  private binOf: Uint8Array | null = null;
  private leak = new Float32Array(RESERVOIR_BINS);
  private tauSec = Float32Array.from({ length: RESERVOIR_BINS }, (_, k) => 0.2 * Math.pow(25, k / (RESERVOIR_BINS - 1)));
  private lastTickMs = 0;

  // ledger / pricing state
  private throttleLatched = false;
  private lastPriceMs = 0;
  private cachedFair = 0;
  private lastMid = 0;
  private lastMidMs = 0;
  private midMoveBps = 0;

  constructor() {
    void this.initTrainedReadout();
    this.connectLiveL2Orderbook();
  }

  private async initTrainedReadout() {
    try {
      const res = await fetch('/models/heston_readout_weights.json');
      if (res.ok) {
        const data = await res.json();
        const flat: number[] = [];
        for (let r = 0; r < RESERVOIR_BINS; r++) for (let c = 0; c < 3; c++) flat.push(data.weights[r][c]);
        this.ridgeWeights = new Float32Array(flat);
        this.ridgeBias = new Float32Array(data.bias);
        const m = data.metrics ?? {};
        const src = data.trained_on ?? 'unknown';
        const r2 = m.test_r2 ? `test R² vol=${m.test_r2[0]?.toFixed?.(2)}` : `R²=${m.r2_score}`;
        this.onnxStatus = `RIDGE ${RESERVOIR_BINS}x3 (trained on ${src}; ${r2})`;
      } else {
        this.onnxStatus = 'READOUT WEIGHTS MISSING: run npm run train:onnx';
      }
    } catch {
      this.onnxStatus = 'READOUT WEIGHTS UNAVAILABLE (default bias only)';
    }

    try {
      ort.env.wasm.wasmPaths = '/ort/';
      ort.env.wasm.numThreads = 1;
      this.onnxSession = await ort.InferenceSession.create('/models/heston_readout.onnx', { executionProviders: ['wasm'] });
      this.onnxStatus += ' + ONNX WASM ACTIVE';
    } catch {
      // JS linear readout computes the same 64x3 product; ONNX is optional.
    }
  }

  private connectLiveL2Orderbook() {
    const offline = () => { this.feedLive = false; this.l2FeedStatus = 'OFFLINE (synthetic random walk)'; };
    try {
      const ws = new WebSocket(L2_URL);
      ws.onopen = () => { this.l2FeedStatus = 'LIVE WS (BTCUSDT depth20@100ms)'; };
      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          const bids: string[][] = msg.bids, asks: string[][] = msg.asks;
          if (!bids?.length || !asks?.length) return;
          const bestBid = parseFloat(bids[0][0]), bestAsk = parseFloat(asks[0][0]);
          if (!(bestBid > 0 && bestAsk > 0)) return;
          let bidQty = 0, askQty = 0;
          for (let i = 0; i < Math.min(BOOK_LEVELS, bids.length); i++) bidQty += parseFloat(bids[i][1]);
          for (let i = 0; i < Math.min(BOOK_LEVELS, asks.length); i++) askQty += parseFloat(asks[i][1]);
          const mid = 0.5 * (bestBid + bestAsk);
          const now = performance.now();
          if (this.lastMid > 0 && now - this.lastMidMs >= 1000) {
            this.midMoveBps = Math.abs((mid / this.lastMid - 1) * 1e4);
            this.lastMid = mid; this.lastMidMs = now;
          } else if (this.lastMid === 0) { this.lastMid = mid; this.lastMidMs = now; }
          this.spotPrice = mid;
          this.bidAskSpreadBps = Math.max(0.01, ((bestAsk - bestBid) / mid) * 1e4);
          this.imbalance = (bidQty - askQty) / Math.max(bidQty + askQty, 1e-9);
          // composite book-pressure drive (a true spread velocity is ~0 on a 1-tick-wide BTCUSDT book)
          this.spreadVelocity = Math.max(0.25, Math.min(3.5, 1.25 + 1.2 * this.imbalance + 0.15 * this.midMoveBps));
          this.feedLive = true;
        } catch { /* ignore malformed frame */ }
      };
      ws.onerror = offline;
      ws.onclose = offline;
    } catch { offline(); }
  }

  public isFeedLive(): boolean { return this.feedLive; }

  public evaluateTick(neurons: Float32Array, neuronCount: number, floatsPerNeuron: number, laserWatts: number, nowMs: number): HestonTelemetry {
    const dtSec = this.lastTickMs ? Math.max(0.01, Math.min(1, (nowMs - this.lastTickMs) / 1000)) : 0.1;
    this.lastTickMs = nowMs;

    // 1. Static spatial binning (computed once positions exist)
    if (!this.binOf && (neurons[F.X] !== 0 || neurons[floatsPerNeuron + F.X] !== 0)) {
      this.binOf = new Uint8Array(neuronCount);
      for (let i = 0; i < neuronCount; i++) {
        const b = i * floatsPerNeuron;
        this.binOf[i] = spatialBin(neurons[b + F.X], neurons[b + F.Y], neurons[b + F.Z]);
      }
    }

    // 2. Pool neurons -> 64 bins; aggregate telemetry in the same pass
    const sum = new Float32Array(RESERVOIR_BINS);
    const cnt = new Uint32Array(RESERVOIR_BINS);
    let sumV = 0, sumAtp = 0, maxTemp = 37.0;
    for (let i = 0; i < neuronCount; i++) {
      const b = i * floatsPerNeuron;
      const v = neurons[b + F.VM];
      sumV += v;
      sumAtp += neurons[b + F.ATP];
      const t = neurons[b + F.TEMP];
      if (t > maxTemp) maxTemp = t;
      if (this.binOf) {
        const k = this.binOf[i];
        sum[k] += (v + 65.0) / 50.0 + neurons[b + F.SPIKE] * 2.0;
        cnt[k]++;
      }
    }
    const meanVm = sumV / Math.max(neuronCount, 1);
    const meanAtp = sumAtp / Math.max(neuronCount, 1);

    const reservoirState = new Float32Array(RESERVOIR_BINS);
    for (let k = 0; k < RESERVOIR_BINS; k++) {
      const x = cnt[k] ? Math.tanh(sum[k] / cnt[k]) : 0;
      const a = 1 - Math.exp(-dtSec / this.tauSec[k]);     // per-bin time constant => fading memory
      this.leak[k] += a * (x - this.leak[k]);
      reservoirState[k] = this.leak[k];
    }

    // 3. Readout: ONNX result if available (one tick stale), else synchronous JS product
    if (this.onnxSession && !this.isInferring) {
      this.isInferring = true;
      this.onnxSession
        .run({ reservoir_state: new ort.Tensor('float32', reservoirState.slice(), [1, RESERVOIR_BINS]) })
        .then((res) => { if (res.heston_out?.data) this.lastAsyncOutput = res.heston_out.data as Float32Array; })
        .catch(() => { this.onnxSession = null; })
        .finally(() => { this.isInferring = false; });
    }
    let volHat = this.ridgeBias[0], vovHat = this.ridgeBias[1], alphaBps = this.ridgeBias[2];
    if (this.lastAsyncOutput && this.lastAsyncOutput.length >= 3) {
      [volHat, vovHat, alphaBps] = [this.lastAsyncOutput[0], this.lastAsyncOutput[1], this.lastAsyncOutput[2]];
    } else {
      for (let k = 0; k < RESERVOIR_BINS; k++) {
        const a = reservoirState[k];
        volHat += a * this.ridgeWeights[k * 3];
        vovHat += a * this.ridgeWeights[k * 3 + 1];
        alphaBps += a * this.ridgeWeights[k * 3 + 2];
      }
    }

    // 4. Variance filter with Feller-safe sigma (full-truncation: variance never goes negative)
    const feller2 = 2 * this.kappa * this.theta;
    const sigmaMax = Math.sqrt(feller2) * 0.99;
    this.sigma = Math.max(0.15, Math.min(sigmaMax, Math.abs(vovHat)));
    const targetVar = Math.max(0.05, Math.min(3.0, Math.abs(volHat))) ** 2;
    const gain = 1 - Math.exp(-dtSec / 5.0);
    const dtYears = dtSec / SECONDS_PER_YEAR;
    const prior = this.kappa * (this.theta - this.variance) * dtYears; // negligible at tick scale (by design)
    this.variance = Math.max(1e-4, Math.min(4.0, this.variance + prior + gain * (targetVar - this.variance)));
    const impliedVolatility = Math.sqrt(this.variance);

    if (!this.feedLive) {
      // offline: random walk with the filter's own variance (clearly labelled synthetic)
      this.spotPrice *= Math.exp(-0.5 * this.variance * dtYears + Math.sqrt(this.variance * dtYears) * gaussian());
      this.spreadVelocity = 1.15 + 0.45 * Math.sin(nowMs / 1000 * 0.7);
      this.imbalance = 0.5 * Math.sin(nowMs / 1000 * 0.31);
    }

    // 5. Real Heston price (ATM, 30d), refreshed at 1 Hz
    if (nowMs - this.lastPriceMs > 1000) {
      this.cachedFair = hestonCallPrice({
        S: this.spotPrice, K: this.spotPrice, tau: 30 / 365, r: this.r,
        v0: this.variance, kappa: this.kappa, theta: this.theta, sigma: this.sigma, rho: this.rho,
      });
      this.lastPriceMs = nowMs;
    }

    // 6. Node 7 toy ledger, everything in bps, per 1-second decision window
    const glucoseDeficit = Math.max(0, LEDGER.ATP_REST_MM - meanAtp);
    const ledgerCostBps = laserWatts * 1.0 * LEDGER.BPS_PER_JOULE + glucoseDeficit * LEDGER.BPS_PER_MM_ATP_DEFICIT;
    const edgeBps = Math.abs(alphaBps);
    let reason = '';
    if (maxTemp > 38.5) reason = 'PHOTOTHERMAL LIMIT';
    else if (meanAtp < 1.0) reason = 'ATP DEPLETION';
    else if (this.throttleLatched ? ledgerCostBps > 0.8 * edgeBps : ledgerCostBps > edgeBps) reason = 'COST > EDGE';
    this.throttleLatched = reason === 'COST > EDGE';
    const throttled = reason !== '';

    const netBps = Math.max(0.01, edgeBps - ledgerCostBps);
    const upiSettlementUri = `upi://pay?pa=${DEMO_VPA}&pn=SYNAPSE-QUANTUM-DEMO&am=${netBps.toFixed(2)}&cu=INR&tn=DEMO_NOT_A_REAL_SETTLEMENT`;

    return {
      spotPrice: this.spotPrice, bidAskSpreadBps: this.bidAskSpreadBps, spreadVelocity: this.spreadVelocity,
      imbalance: this.imbalance, feedLive: this.feedLive, instantVariance: this.variance, impliedVolatility,
      volOfVol: this.sigma, feller: feller2 > this.sigma * this.sigma, hestonOptionFairValue: this.cachedFair,
      predictedAlphaBps: alphaBps, ledgerCostBps, throttled, throttleReason: reason, upiSettlementUri,
      onnxStatus: this.onnxStatus, l2FeedStatus: this.l2FeedStatus, reservoirState, meanVm, meanAtp, maxTempC: maxTemp,
    };
  }
}

function gaussian(): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
