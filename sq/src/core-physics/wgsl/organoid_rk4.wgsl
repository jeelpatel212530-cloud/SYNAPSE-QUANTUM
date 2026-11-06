// ============================================================================
// SYNAPSE-QUANTUM: WebGPU 4th-Order Runge-Kutta (RK4) Cortical Organoid Shader
// Integrates 10,240 Hodgkin-Huxley neurons coupled to a 4-state ChR2 photocycle,
// two-photon Henyey-Greenstein ballistic scattering exp(-2*depth/l_s), axial
// focal-plane confinement, 32x32 Gerchberg-Saxton holographic SLM gain mask,
// Michaelis-Menten intracellular ATP/glucose kinetics, and bio-heat relaxation.
// ============================================================================

struct SimParams {
  dt: f32,
  p_laser: f32,
  na: f32,
  wavelength: f32,
  tau_pulse: f32,
  f_rep: f32,
  l_s: f32,
  market_modulation: f32,
  slm_focal_z: f32,
  organoid_radius: f32,
  slm_grid: f32,
  _pad: f32,
};

struct Neuron {
  x: f32,
  y: f32,
  z: f32,
  vm: f32,
  m: f32,
  h: f32,
  n: f32,
  c1: f32,
  o1: f32,
  o2: f32,
  c2: f32,
  atp: f32,
  glc: f32,
  ichr2: f32,
  spike: f32,
  temp: f32,
};

struct HHState {
  v: f32,
  m: f32,
  h: f32,
  n: f32,
};

struct HHDeriv {
  dv: f32,
  dm: f32,
  dh: f32,
  dn: f32,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> neurons: array<Neuron>;
@group(0) @binding(2) var<storage, read> slm_mask: array<f32>;

const HH_DT: f32 = 0.05;
const SUBSTEPS: u32 = 20u;
const AXIAL_SIGMA_UM: f32 = 60.0;
const K_IRR: f32 = 40.0;
const C_KA1: f32 = 0.5;
const C_KA2: f32 = 0.1;
const C_KD1: f32 = 0.1;
const C_KD2: f32 = 0.05;
const C_E12: f32 = 0.011;
const C_E21: f32 = 0.008;
const C_KR: f32 = 0.004;
const G_CHR2: f32 = 0.4;
const DEG_C_PER_MW: f32 = 0.019;
const ATP_REST: f32 = 3.2;
const GLC_REST: f32 = 4.5;

fn hash01(idx: u32) -> f32 {
  let h: u32 = ((idx + 1u) * 2654435761u) % 100000u;
  return f32(h) / 100000.0;
}

fn alpha_m(v: f32) -> f32 {
  let x: f32 = v + 40.0;
  if (abs(x) < 1e-5) {
    return 1.0;
  }
  return (0.1 * x) / (1.0 - exp(-x / 10.0));
}

fn beta_m(v: f32) -> f32 {
  return 4.0 * exp(-(v + 65.0) / 18.0);
}

fn alpha_h(v: f32) -> f32 {
  return 0.07 * exp(-(v + 65.0) / 20.0);
}

fn beta_h(v: f32) -> f32 {
  return 1.0 / (1.0 + exp(-(v + 35.0) / 10.0));
}

fn alpha_n(v: f32) -> f32 {
  let x: f32 = v + 55.0;
  if (abs(x) < 1e-5) {
    return 0.1;
  }
  return (0.01 * x) / (1.0 - exp(-x / 10.0));
}

fn beta_n(v: f32) -> f32 {
  return 0.125 * exp(-(v + 65.0) / 80.0);
}

fn eval_hh_deriv(
  st: HHState,
  e_na: f32,
  e_k: f32,
  g_photo: f32,
  tonic: f32,
  phi: f32
) -> HHDeriv {
  let m_clamped: f32 = clamp(st.m, 0.0, 1.0);
  let h_clamped: f32 = clamp(st.h, 0.0, 1.0);
  let n_clamped: f32 = clamp(st.n, 0.0, 1.0);

  let m3: f32 = m_clamped * m_clamped * m_clamped;
  let n4: f32 = n_clamped * n_clamped * n_clamped * n_clamped;

  let i_na: f32 = 120.0 * m3 * h_clamped * (st.v - e_na);
  let i_k: f32 = 36.0 * n4 * (st.v - e_k);
  let i_l: f32 = 0.3 * (st.v + 54.387);
  let i_photo: f32 = -g_photo * (st.v - 0.0);

  let dv: f32 = -i_na - i_k - i_l + i_photo + tonic;

  let am: f32 = alpha_m(st.v);
  let bm: f32 = beta_m(st.v);
  let ah: f32 = alpha_h(st.v);
  let bh: f32 = beta_h(st.v);
  let an: f32 = alpha_n(st.v);
  let bn: f32 = beta_n(st.v);

  let dm: f32 = phi * (am * (1.0 - m_clamped) - bm * m_clamped);
  let dh: f32 = phi * (ah * (1.0 - h_clamped) - bh * h_clamped);
  let dn: f32 = phi * (an * (1.0 - n_clamped) - bn * n_clamped);

  return HHDeriv(dv, dm, dh, dn);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx: u32 = gid.x;
  let total_neurons: u32 = arrayLength(&neurons);
  if (idx >= total_neurons) {
    return;
  }

  var cell: Neuron = neurons[idx];

  let radius: f32 = select(360.0, params.organoid_radius, params.organoid_radius > 1.0);
  let grid_f: f32 = select(32.0, params.slm_grid, params.slm_grid >= 4.0);
  let grid_n: u32 = u32(grid_f);

  // Two-photon ballistic scattering depth below top surface (z = -radius)
  let depth: f32 = max(0.0, cell.z + radius);
  let dz: f32 = cell.z - params.slm_focal_z;
  let axial: f32 = exp(-(dz * dz) / (2.0 * AXIAL_SIGMA_UM * AXIAL_SIGMA_UM));
  let ls_safe: f32 = max(params.l_s, 10.0);
  let atten: f32 = exp((-2.0 * depth) / ls_safe);

  // Map (x, y) into 32x32 Gerchberg-Saxton SLM focal intensity grid
  let norm_x: f32 = clamp((cell.x + radius) / (2.0 * radius), 0.0, 0.9999);
  let norm_y: f32 = clamp((cell.y + radius) / (2.0 * radius), 0.0, 0.9999);
  let ix: u32 = min(grid_n - 1u, u32(floor(norm_x * grid_f)));
  let iy: u32 = min(grid_n - 1u, u32(floor(norm_y * grid_f)));
  let slm_idx: u32 = min(iy * grid_n + ix, arrayLength(&slm_mask) - 1u);
  let slm_gain: f32 = 0.3 + 0.7 * slm_mask[slm_idx];

  // Optical pulse & NA factor (normalised to 1.0 at reference NA=1.05, lambda=920nm, tau=140fs, f_rep=80MHz)
  let duty_norm: f32 = clamp((140.0 / max(params.tau_pulse, 20.0)) * (80.0 / max(params.f_rep, 10.0)), 0.5, 2.0);
  let na_norm: f32 = clamp((params.na / 1.05) * (920.0 / max(params.wavelength, 400.0)), 0.5, 2.0);
  let p_ratio: f32 = (params.p_laser * params.market_modulation) / 35.0;

  var irr: f32 = K_IRR * p_ratio * p_ratio * atten * axial * slm_gain * duty_norm * na_norm;
  irr = 4.0 * tanh(irr / 4.0);
  let ka1: f32 = C_KA1 * irr;
  let ka2: f32 = C_KA2 * irr;

  // Parenchyma bio-heat relaxation & Q10 = 3 kinetic scaling
  let t_target: f32 = 37.0 + DEG_C_PER_MW * params.p_laser * exp(-depth / ls_safe);
  cell.temp = cell.temp + (t_target - cell.temp) * 0.02;
  let phi: f32 = pow(3.0, (cell.temp - 37.0) / 10.0);
  let tonic: f32 = (hash01(idx) - 0.5) * 1.0;

  let pump_rest: f32 = ATP_REST / (0.8 + ATP_REST);
  let step_h: f32 = select(HH_DT, clamp(params.dt * 2.5, 0.01, 0.05), params.dt > 0.0);

  var spiked: bool = false;
  var i_photo_last: f32 = 0.0;

  for (var s: u32 = 0u; s < SUBSTEPS; s = s + 1u) {
    // 1. 4-state ChR2 photocycle update
    let dc1: f32 = -ka1 * cell.c1 + C_KD1 * cell.o1 + C_KR * cell.c2;
    let do1: f32 = ka1 * cell.c1 - (C_KD1 + C_E12) * cell.o1 + C_E21 * cell.o2;
    let do2: f32 = ka2 * cell.c2 - (C_KD2 + C_E21) * cell.o2 + C_E12 * cell.o1;
    let dc2: f32 = -(ka2 + C_KR) * cell.c2 + C_KD2 * cell.o2;

    cell.c1 = max(0.0, cell.c1 + step_h * dc1);
    cell.o1 = max(0.0, cell.o1 + step_h * do1);
    cell.o2 = max(0.0, cell.o2 + step_h * do2);
    cell.c2 = max(0.0, cell.c2 + step_h * dc2);
    let c_sum: f32 = max(1e-6, cell.c1 + cell.o1 + cell.o2 + cell.c2);
    cell.c1 = cell.c1 / c_sum;
    cell.o1 = cell.o1 / c_sum;
    cell.o2 = cell.o2 / c_sum;
    cell.c2 = cell.c2 / c_sum;

    // 2. ATP-dependent Na+/K+ pump reversal potentials
    let pump: f32 = min(1.0, (cell.atp / (0.8 + cell.atp)) / pump_rest);
    let e_na: f32 = 50.0 * pump;
    let e_k: f32 = -77.0 * pump;
    let g_photo: f32 = G_CHR2 * (cell.o1 + 0.1 * cell.o2);

    // 3. 4th-Order Runge-Kutta (RK4) integration of Hodgkin-Huxley state (V, m, h, n)
    let st0: HHState = HHState(cell.vm, cell.m, cell.h, cell.n);
    let k1: HHDeriv = eval_hh_deriv(st0, e_na, e_k, g_photo, tonic, phi);

    let st1: HHState = HHState(
      st0.v + 0.5 * step_h * k1.dv,
      clamp(st0.m + 0.5 * step_h * k1.dm, 0.0, 1.0),
      clamp(st0.h + 0.5 * step_h * k1.dh, 0.0, 1.0),
      clamp(st0.n + 0.5 * step_h * k1.dn, 0.0, 1.0)
    );
    let k2: HHDeriv = eval_hh_deriv(st1, e_na, e_k, g_photo, tonic, phi);

    let st2: HHState = HHState(
      st0.v + 0.5 * step_h * k2.dv,
      clamp(st0.m + 0.5 * step_h * k2.dm, 0.0, 1.0),
      clamp(st0.h + 0.5 * step_h * k2.dh, 0.0, 1.0),
      clamp(st0.n + 0.5 * step_h * k2.dn, 0.0, 1.0)
    );
    let k3: HHDeriv = eval_hh_deriv(st2, e_na, e_k, g_photo, tonic, phi);

    let st3: HHState = HHState(
      st0.v + step_h * k3.dv,
      clamp(st0.m + step_h * k3.dm, 0.0, 1.0),
      clamp(st0.h + step_h * k3.dh, 0.0, 1.0),
      clamp(st0.n + step_h * k3.dn, 0.0, 1.0)
    );
    let k4: HHDeriv = eval_hh_deriv(st3, e_na, e_k, g_photo, tonic, phi);

    let v_next: f32 = clamp(
      st0.v + (step_h / 6.0) * (k1.dv + 2.0 * k2.dv + 2.0 * k3.dv + k4.dv),
      -100.0,
      60.0
    );
    if (cell.vm < 0.0 && v_next >= 0.0) {
      spiked = true;
    }
    cell.vm = v_next;
    cell.m = clamp(st0.m + (step_h / 6.0) * (k1.dm + 2.0 * k2.dm + 2.0 * k3.dm + k4.dm), 0.0, 1.0);
    cell.h = clamp(st0.h + (step_h / 6.0) * (k1.dh + 2.0 * k2.dh + 2.0 * k3.dh + k4.dh), 0.0, 1.0);
    cell.n = clamp(st0.n + (step_h / 6.0) * (k1.dn + 2.0 * k2.dn + 2.0 * k3.dn + k4.dn), 0.0, 1.0);

    i_photo_last = -g_photo * (cell.vm - 0.0);

    // 4. Michaelis-Menten intracellular ATP & glucose kinetics
    let i_na_mag: f32 = abs(120.0 * cell.m * cell.m * cell.m * cell.h * (cell.vm - e_na));
    let prod: f32 = (0.004 * (cell.glc / (1.0 + cell.glc))) / 0.8182;
    let cons: f32 = 0.004 * (cell.atp / ATP_REST) + 1e-4 * i_na_mag;
    cell.atp = clamp(cell.atp + step_h * (prod - cons), 0.05, 4.0);
    cell.glc = max(0.1, cell.glc + step_h * (-0.02 * prod + 1e-3 * (GLC_REST - cell.glc)));
  }

  cell.ichr2 = i_photo_last;
  cell.spike = select(0.0, 1.0, spiked);
  neurons[idx] = cell;
}
