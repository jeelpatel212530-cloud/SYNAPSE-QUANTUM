// ============================================================================
// SYNAPSE-QUANTUM: Decoupled WebGPU RK4 Physics + WebGL2 3D Volumetric Worker
// Orchestrates 10,240 neurons, a real Gerchberg-Saxton SLM hologram, and 60 FPS
// 3D volumetric dendritic filament & Gaussian soma rendering (Nodes 2, 8, 13).
// ============================================================================

import shaderSource from '../wgsl/organoid_rk4.wgsl?raw';
import {
  NEURON_COUNT, FLOATS_PER_NEURON, DENDRITE_EDGE_COUNT, ORGANOID_RADIUS_UM, SLM_GRID, F,
} from '../../constants';
import { gerchbergSaxton, buildTargetAmplitude } from '../../optics/gerchberg_saxton';
import { mulberry32 } from '../../util/rng';

let sharedFloatView: Float32Array | null = null;
let useSnapshots = false; // true when SharedArrayBuffer is unavailable: post copies instead
let offscreenCanvas: OffscreenCanvas | null = null;
let gl: WebGL2RenderingContext | null = null;
let ctx2d: OffscreenCanvasRenderingContext2D | null = null;

let somaProgram: WebGLProgram | null = null;
let filamentProgram: WebGLProgram | null = null;
let somaVao: WebGLVertexArrayObject | null = null;
let somaBuffer: WebGLBuffer | null = null;
let filamentVao: WebGLVertexArrayObject | null = null;
let filamentBuffer: WebGLBuffer | null = null;

const somaRenderData = new Float32Array(NEURON_COUNT * 6);
const filamentRenderData = new Float32Array(DENDRITE_EDGE_COUNT * 2 * 5);
const dendritePairs = new Uint32Array(DENDRITE_EDGE_COUNT * 2);

let gpuDevice: GPUDevice | null = null;
let computePipeline: GPUComputePipeline | null = null;
let neuronBuffer: GPUBuffer | null = null;
let stagingBuffer: GPUBuffer | null = null;
let paramsBuffer: GPUBuffer | null = null;
let slmBuffer: GPUBuffer | null = null;
let bindGroup: GPUBindGroup | null = null;

let isRunning = false;
let throttledByLedger = false;

let simParams = {
  dt: 0.02,
  p_laser: 35.0,
  na: 1.05,
  wavelength: 920.0,
  tau_pulse: 140.0,
  f_rep: 80.0,
  l_s: 180.0,
  market_modulation: 1.25,
  market_imbalance: 0.0,
  palette_id: 0,
  camera_yaw: 0.4,
  camera_pitch: 0.25,
  camera_zoom: 680.0,
  slm_focal_z: 0.0,
  show_dendrites: 1,
};

// ---- Gerchberg-Saxton hologram (real phase retrieval, see optics/gerchberg_saxton.ts) ----
let slmMask: Float32Array = new Float32Array(SLM_GRID * SLM_GRID).fill(0.5);
const gpuParamsData = new Float32Array(12);
let lastPhase: Float64Array | undefined;
let gsEfficiency = 0;
let gsUniformity = 0;

function rebuildHologram() {
  const target = buildTargetAmplitude(simParams.market_modulation, simParams.market_imbalance);
  const res = gerchbergSaxton(target, 10, lastPhase); // warm-started from previous phase
  lastPhase = res.phase;
  slmMask = res.mask;
  gsEfficiency = res.efficiency;
  gsUniformity = res.uniformity;
  if (gpuDevice && slmBuffer) gpuDevice.queue.writeBuffer(slmBuffer, 0, slmMask.buffer as ArrayBuffer, slmMask.byteOffset, slmMask.byteLength);
}

// ---- Organoid seeding (seeded PRNG => reproducible runs) ----
function seedOrganoidPopulation(view: Float32Array) {
  const rnd = mulberry32(1337);
  for (let i = 0; i < NEURON_COUNT; i++) {
    const base = i * FLOATS_PER_NEURON;
    const theta = rnd() * 2.0 * Math.PI;
    const phi = Math.acos(2.0 * rnd() - 1.0);
    const r = Math.cbrt(rnd()) * ORGANOID_RADIUS_UM;

    view[base + F.X] = r * Math.sin(phi) * Math.cos(theta);
    view[base + F.Y] = r * Math.sin(phi) * Math.sin(theta);
    view[base + F.Z] = r * Math.cos(phi);
    view[base + F.VM] = -65.0 + (rnd() - 0.5) * 6.0;
    view[base + F.M] = 0.0529;
    view[base + F.H] = 0.5961;
    view[base + F.N] = 0.3177;
    view[base + F.C1] = 1.0;
    view[base + F.O1] = 0.0;
    view[base + F.O2] = 0.0;
    view[base + F.C2] = 0.0;
    view[base + F.ATP] = 3.2;
    view[base + F.GLC] = 4.5;
    view[base + F.ICHR2] = 0.0;
    view[base + F.SPIKE] = 0.0;
    view[base + F.TEMP] = 37.0;
  }

  // Nearest-neighbour dendritic tracts (min distance 20 um)
  let edgeIdx = 0;
  for (let i = 0; i < NEURON_COUNT && edgeIdx < DENDRITE_EDGE_COUNT; i += 5) {
    const b1 = i * FLOATS_PER_NEURON;
    const x1 = view[b1], y1 = view[b1 + 1], z1 = view[b1 + 2];
    let bestJ = (i + 17) % NEURON_COUNT;
    let bestDist2 = 1e12;
    for (let probe = 1; probe < 40; probe++) {
      const candidate = (i + probe * 13) % NEURON_COUNT;
      const b2 = candidate * FLOATS_PER_NEURON;
      const dx = view[b2] - x1, dy = view[b2 + 1] - y1, dz = view[b2 + 2] - z1;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > 400.0 && d2 < bestDist2) { bestDist2 = d2; bestJ = candidate; }
    }
    dendritePairs[edgeIdx * 2] = i;
    dendritePairs[edgeIdx * 2 + 1] = bestJ;
    edgeIdx++;
  }
}

function compileGLSLProgram(glCtx: WebGL2RenderingContext, vsSrc: string, fsSrc: string): WebGLProgram | null {
  const mk = (type: number, src: string, label: string) => {
    const sh = glCtx.createShader(type)!;
    glCtx.shaderSource(sh, src);
    glCtx.compileShader(sh);
    if (!glCtx.getShaderParameter(sh, glCtx.COMPILE_STATUS)) {
      console.error(`${label} compile error:`, glCtx.getShaderInfoLog(sh));
      return null;
    }
    return sh;
  };
  const vs = mk(glCtx.VERTEX_SHADER, vsSrc, 'VS');
  const fs = mk(glCtx.FRAGMENT_SHADER, fsSrc, 'FS');
  if (!vs || !fs) return null;
  const prog = glCtx.createProgram()!;
  glCtx.attachShader(prog, vs);
  glCtx.attachShader(prog, fs);
  glCtx.linkProgram(prog);
  if (!glCtx.getProgramParameter(prog, glCtx.LINK_STATUS)) {
    console.error('Program link error:', glCtx.getProgramInfoLog(prog));
    return null;
  }
  return prog;
}

// Initialize Hardware-Accelerated WebGL2 3D Volumetric Point-Cloud & Dendritic Pipeline
function initWebGL2VolumetricPipeline(canvas: OffscreenCanvas): boolean {
  const context = canvas.getContext('webgl2', { antialias: true, alpha: false });
  if (!context) return false;
  gl = context;

  // 1. 3D Gaussian Soma Volumetric Point-Cloud Shaders (GLSL ES 3.00)
  const somaVS = `#version 300 es
    precision highp float;
    layout(location = 0) in vec3 a_pos;
    layout(location = 1) in float a_vm;
    layout(location = 2) in float a_spike;
    layout(location = 3) in float a_ichr2;

    uniform float u_yaw;
    uniform float u_pitch;
    uniform float u_zoom;
    uniform float u_aspect;
    uniform float u_palette;
    uniform float u_focalZ;

    out vec3 v_color;
    out float v_spike;
    out float v_focalGlow;

    vec3 fluorophorePalette(float t, float pal, float spike) {
      t = clamp(t, 0.0, 1.0);
      if (spike > 0.5) return vec3(0.1, 1.0, 0.9);
      if (pal < 0.5) {
        // Viridis
        return vec3(0.267 + t * 0.725, 0.004 + t * 0.902, 0.329 + sin(t * 3.14159) * 0.55 - t * 0.18);
      } else if (pal < 1.5) {
        // GCaMP6f Green
        return vec3(0.04 + 0.35 * t, 0.18 + 0.82 * pow(t, 0.75), 0.08 + 0.28 * t);
      } else if (pal < 2.5) {
        // mCherry Red-Magenta
        return vec3(0.28 + 0.72 * t, 0.04 + 0.25 * t * t, 0.20 + 0.58 * t);
      } else {
        // DAPI Blue-Cyan
        return vec3(0.08 + 0.32 * t, 0.22 + 0.65 * t, 0.48 + 0.52 * pow(t, 0.7));
      }
    }

    void main() {
      float cy = cos(u_yaw), sy = sin(u_yaw);
      float cp = cos(u_pitch), sp = sin(u_pitch);

      // Rotate around Y then X
      vec3 p1 = vec3(a_pos.x * cy - a_pos.z * sy, a_pos.y, a_pos.x * sy + a_pos.z * cy);
      vec3 p2 = vec3(p1.x, p1.y * cp - p1.z * sp, p1.y * sp + p1.z * cp);

      float eyeZ = p2.z + u_zoom;
      float projScale = 1.65 / max(eyeZ, 80.0);

      gl_Position = vec4((p2.x * projScale) / u_aspect, p2.y * projScale, clamp((eyeZ - 100.0) / 1500.0, -0.99, 0.99), 1.0);

      float normV = (a_vm + 80.0) / 130.0;
      float focalDist = abs(a_pos.z - u_focalZ);
      v_focalGlow = exp(-focalDist / 42.0);
      v_color = fluorophorePalette(normV, u_palette, a_spike);
      v_spike = a_spike;
      gl_PointSize = clamp((3400.0 / max(eyeZ, 120.0)) * (1.0 + 0.75 * a_spike + 0.35 * v_focalGlow), 2.0, 14.0);
    }
  `;

  const somaFS = `#version 300 es
    precision highp float;
    in vec3 v_color;
    in float v_spike;
    in float v_focalGlow;
    out vec4 outColor;

    void main() {
      vec2 coord = gl_PointCoord * 2.0 - 1.0;
      float r2 = dot(coord, coord);
      if (r2 > 1.0) discard;

      // Volumetric 3D Gaussian soma falloff with SLM two-photon halo
      float gauss = exp(-r2 * 2.6);
      vec3 col = v_color + vec3(0.0, 0.28, 0.42) * v_focalGlow;
      float alpha = gauss * (0.65 + 0.35 * v_spike);
      outColor = vec4(col * (1.0 + 0.55 * v_spike), alpha);
    }
  `;

  // 2. Volumetric Dendritic Filaments & SLM Holographic Ring Shader
  const filamentVS = `#version 300 es
    precision highp float;
    layout(location = 0) in vec3 a_pos;
    layout(location = 1) in float a_vm;
    layout(location = 2) in float a_spike;

    uniform float u_yaw;
    uniform float u_pitch;
    uniform float u_zoom;
    uniform float u_aspect;
    uniform float u_palette;

    out vec4 v_lineColor;

    void main() {
      float cy = cos(u_yaw), sy = sin(u_yaw);
      float cp = cos(u_pitch), sp = sin(u_pitch);
      vec3 p1 = vec3(a_pos.x * cy - a_pos.z * sy, a_pos.y, a_pos.x * sy + a_pos.z * cy);
      vec3 p2 = vec3(p1.x, p1.y * cp - p1.z * sp, p1.y * sp + p1.z * cp);

      float eyeZ = p2.z + u_zoom;
      float projScale = 1.65 / max(eyeZ, 80.0);
      gl_Position = vec4((p2.x * projScale) / u_aspect, p2.y * projScale, clamp((eyeZ - 100.0) / 1500.0, -0.99, 0.99), 1.0);

      float normV = clamp((a_vm + 80.0) / 130.0, 0.0, 1.0);
      vec3 baseCol = mix(vec3(0.08, 0.28, 0.52), vec3(0.0, 0.95, 0.82), normV * 0.7 + a_spike * 0.5);
      float alpha = 0.14 + 0.38 * a_spike + 0.15 * normV;
      v_lineColor = vec4(baseCol, clamp(alpha, 0.08, 0.75));
    }
  `;

  const filamentFS = `#version 300 es
    precision highp float;
    in vec4 v_lineColor;
    out vec4 outColor;
    void main() {
      outColor = v_lineColor;
    }
  `;

  somaProgram = compileGLSLProgram(gl, somaVS, somaFS);
  filamentProgram = compileGLSLProgram(gl, filamentVS, filamentFS);
  if (!somaProgram || !filamentProgram) return false;

  // Configure Soma VAO
  somaVao = gl.createVertexArray();
  somaBuffer = gl.createBuffer();
  gl.bindVertexArray(somaVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, somaBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, somaRenderData.byteLength, gl.DYNAMIC_DRAW);

  const strideSoma = 6 * 4;
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, strideSoma, 0);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 1, gl.FLOAT, false, strideSoma, 12);
  gl.enableVertexAttribArray(2);
  gl.vertexAttribPointer(2, 1, gl.FLOAT, false, strideSoma, 16);
  gl.enableVertexAttribArray(3);
  gl.vertexAttribPointer(3, 1, gl.FLOAT, false, strideSoma, 20);

  // Configure Dendritic Filament VAO
  filamentVao = gl.createVertexArray();
  filamentBuffer = gl.createBuffer();
  gl.bindVertexArray(filamentVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, filamentBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, filamentRenderData.byteLength, gl.DYNAMIC_DRAW);

  const strideFil = 5 * 4;
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, strideFil, 0);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 1, gl.FLOAT, false, strideFil, 12);
  gl.enableVertexAttribArray(2);
  gl.vertexAttribPointer(2, 1, gl.FLOAT, false, strideFil, 16);

  gl.bindVertexArray(null);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE); // Additive volumetric bioluminescent blending
  return true;
}

// Render 60 FPS WebGL2 3D Volumetric Organoid & Dendritic Connectome
function renderVolumetricWebGL2(view: Float32Array, timeMs: number) {
  if (!gl || !offscreenCanvas || !somaProgram || !filamentProgram) return;

  const w = offscreenCanvas.width;
  const h = offscreenCanvas.height;
  gl.viewport(0, 0, w, h);
  gl.clearColor(0.02, 0.035, 0.065, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  const autoYaw = simParams.camera_yaw + timeMs * 0.00022;
  const aspect = w / Math.max(h, 1);

  // 1. Update & Draw 3D Volumetric Dendritic Filaments
  if (simParams.show_dendrites > 0.5) {
    for (let e = 0; e < DENDRITE_EDGE_COUNT; e++) {
      const n1 = dendritePairs[e * 2] * FLOATS_PER_NEURON;
      const n2 = dendritePairs[e * 2 + 1] * FLOATS_PER_NEURON;
      const dst = e * 10;

      filamentRenderData[dst + 0] = view[n1 + 0];
      filamentRenderData[dst + 1] = view[n1 + 1];
      filamentRenderData[dst + 2] = view[n1 + 2];
      filamentRenderData[dst + 3] = view[n1 + 3];
      filamentRenderData[dst + 4] = view[n1 + 14];

      filamentRenderData[dst + 5] = view[n2 + 0];
      filamentRenderData[dst + 6] = view[n2 + 1];
      filamentRenderData[dst + 7] = view[n2 + 2];
      filamentRenderData[dst + 8] = view[n2 + 3];
      filamentRenderData[dst + 9] = view[n2 + 14];
    }

    gl.useProgram(filamentProgram);
    gl.uniform1f(gl.getUniformLocation(filamentProgram, 'u_yaw'), autoYaw);
    gl.uniform1f(gl.getUniformLocation(filamentProgram, 'u_pitch'), simParams.camera_pitch);
    gl.uniform1f(gl.getUniformLocation(filamentProgram, 'u_zoom'), simParams.camera_zoom);
    gl.uniform1f(gl.getUniformLocation(filamentProgram, 'u_aspect'), aspect);
    gl.uniform1f(gl.getUniformLocation(filamentProgram, 'u_palette'), simParams.palette_id);

    gl.bindVertexArray(filamentVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, filamentBuffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, filamentRenderData);
    gl.drawArrays(gl.LINES, 0, DENDRITE_EDGE_COUNT * 2);
  }

  // 2. Update & Draw 10,240 3D Gaussian Soma Splats
  for (let i = 0; i < NEURON_COUNT; i++) {
    const b = i * FLOATS_PER_NEURON;
    const dst = i * 6;
    somaRenderData[dst + 0] = view[b + 0];
    somaRenderData[dst + 1] = view[b + 1];
    somaRenderData[dst + 2] = view[b + 2];
    somaRenderData[dst + 3] = view[b + 3];
    somaRenderData[dst + 4] = view[b + 14];
    somaRenderData[dst + 5] = view[b + 13];
  }

  gl.useProgram(somaProgram);
  gl.uniform1f(gl.getUniformLocation(somaProgram, 'u_yaw'), autoYaw);
  gl.uniform1f(gl.getUniformLocation(somaProgram, 'u_pitch'), simParams.camera_pitch);
  gl.uniform1f(gl.getUniformLocation(somaProgram, 'u_zoom'), simParams.camera_zoom);
  gl.uniform1f(gl.getUniformLocation(somaProgram, 'u_aspect'), aspect);
  gl.uniform1f(gl.getUniformLocation(somaProgram, 'u_palette'), simParams.palette_id);
  gl.uniform1f(gl.getUniformLocation(somaProgram, 'u_focalZ'), simParams.slm_focal_z);

  gl.bindVertexArray(somaVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, somaBuffer);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, somaRenderData);
  gl.drawArrays(gl.POINTS, 0, NEURON_COUNT);
}

// ---- 2D fallback renderer (used only if WebGL2 is unavailable) ----
function renderFallback2D(view: Float32Array, timeMs: number) {
  if (!ctx2d || !offscreenCanvas) return;
  const w = offscreenCanvas.width, h = offscreenCanvas.height;
  ctx2d.fillStyle = '#050912';
  ctx2d.fillRect(0, 0, w, h);
  const yaw = simParams.camera_yaw + timeMs * 0.00022;
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const scale = (Math.min(w, h) * 0.5) / ORGANOID_RADIUS_UM;
  for (let i = 0; i < NEURON_COUNT; i += 2) {
    const b = i * FLOATS_PER_NEURON;
    const x = view[b + F.X] * cy - view[b + F.Z] * sy;
    const spike = view[b + F.SPIKE] > 0.5;
    const t = Math.max(0, Math.min(1, (view[b + F.VM] + 80) / 130));
    ctx2d.fillStyle = spike ? '#1affe6' : `rgb(${20 + 60 * t},${80 + 150 * t},${140 + 80 * t})`;
    ctx2d.fillRect(w / 2 + x * scale, h / 2 + view[b + F.Y] * scale, spike ? 3 : 2, spike ? 3 : 2);
  }
}

async function initWebGPU(initialView: Float32Array): Promise<boolean> {
  try {
    if (!navigator.gpu) return false;
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return false;
    const device = await adapter.requestDevice();
    device.lost.then((info) => {
      console.error('WebGPU device lost:', info.message);
      gpuDevice = null; // loop falls back to CPU path
    });
    gpuDevice = device;

    const byteLength = initialView.byteLength;
    neuronBuffer = device.createBuffer({
      size: byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(neuronBuffer, 0, initialView.buffer as ArrayBuffer, initialView.byteOffset, byteLength);
    stagingBuffer = device.createBuffer({ size: byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    paramsBuffer = device.createBuffer({ size: gpuParamsData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    slmBuffer = device.createBuffer({ size: slmMask.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(slmBuffer, 0, slmMask.buffer as ArrayBuffer, slmMask.byteOffset, slmMask.byteLength);

    const shaderModule = device.createShaderModule({ code: shaderSource });
    const info = await shaderModule.getCompilationInfo();
    const errs = info.messages.filter((m) => m.type === 'error');
    if (errs.length) {
      console.error('WGSL compile errors:', errs.map((e) => `${e.lineNum}: ${e.message}`).join('\n'));
      gpuDevice = null;
      return false;
    }
    computePipeline = device.createComputePipeline({ layout: 'auto', compute: { module: shaderModule, entryPoint: 'main' } });
    bindGroup = device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: neuronBuffer } },
        { binding: 2, resource: { buffer: slmBuffer } },
      ],
    });
    return true;
  } catch (err) {
    console.error('WebGPU init failed, using CPU physics:', err);
    gpuDevice = null;
    return false;
  }
}

// ============================================================================
// CPU physics path: full Hodgkin-Huxley (gates integrated), 4-state ChR2 photocycle,
// two-photon drive ~ (P*mod)^2 * exp(-2*depth/l_s) with axial confinement around the SLM
// focal plane, heating ~ P*exp(-depth/l_s), Michaelis-Menten ATP/glucose kinetics.
// NOTE: the WGSL shader is a separate implementation; keep both models in sync.
// ============================================================================
const HH_DT = 0.05;       // ms per substep
const SUBSTEPS = 20;      // => 1 ms of biological time per frame (CPU is the fallback path)
const AXIAL_SIGMA_UM = 60;
const K_IRR = 40;         // irradiance scale at P=35 mW, mod=1, depth=0, on focal plane
const C_KA1 = 0.5, C_KA2 = 0.1, C_KD1 = 0.1, C_KD2 = 0.05, C_E12 = 0.011, C_E21 = 0.008, C_KR = 0.004; // /ms
const G_CHR2 = 0.4;       // mS/cm^2
const DEG_C_PER_MW = 0.019;
const ATP_REST = 3.2, GLC_REST = 4.5;

function hash01(i: number): number {
  return ((Math.imul(i + 1, 2654435761) >>> 0) % 100000) / 100000;
}

function stepCPUPhysics(view: Float32Array, laserMw: number): number {
  let spikesThisFrame = 0;
  const R = ORGANOID_RADIUS_UM;
  const N = SLM_GRID;
  const mod = simParams.market_modulation;
  const pRatio = (laserMw * mod) / 35.0;
  const pumpRest = ATP_REST / (0.8 + ATP_REST);

  for (let i = 0; i < NEURON_COUNT; i++) {
    const b = i * FLOATS_PER_NEURON;
    const x = view[b + F.X], y = view[b + F.Y], z = view[b + F.Z];
    let v = view[b + F.VM], m = view[b + F.M], h = view[b + F.H], n = view[b + F.N];
    let c1 = view[b + F.C1], o1 = view[b + F.O1], o2 = view[b + F.O2], c2 = view[b + F.C2];
    let atp = view[b + F.ATP], glc = view[b + F.GLC], temp = view[b + F.TEMP];

    const depth = Math.max(0, z + R);                   // um below the illuminated surface (z = -R)
    const dz = z - simParams.slm_focal_z;
    const axial = Math.exp(-(dz * dz) / (2 * AXIAL_SIGMA_UM * AXIAL_SIGMA_UM));
    const atten = Math.exp((-2.0 * depth) / simParams.l_s);
    const ix = Math.max(0, Math.min(N - 1, Math.floor(((x + R) / (2 * R)) * N)));
    const iy = Math.max(0, Math.min(N - 1, Math.floor(((y + R) / (2 * R)) * N)));
    const slmGain = 0.3 + 0.7 * slmMask[iy * N + ix];
    let irr = K_IRR * pRatio * pRatio * atten * axial * slmGain;
    irr = 4 * Math.tanh(irr / 4);                        // photocycle saturation; keeps Euler stable
    const ka1 = C_KA1 * irr, ka2 = C_KA2 * irr;

    const tTarget = 37.0 + DEG_C_PER_MW * laserMw * Math.exp(-depth / simParams.l_s);
    temp += (tTarget - temp) * 0.02;                      // thermal relaxation per frame
    const phi = Math.pow(3, (temp - 37.0) / 10);          // Q10 = 3 relative to 37 C
    const tonic = (hash01(i) - 0.5) * 1.0;                // per-neuron heterogeneity (uA/cm^2)

    let spiked = false, iPhoto = 0;
    for (let s = 0; s < SUBSTEPS; s++) {
      // ChR2 4-state photocycle (C1 <-> O1 <-> O2 <-> C2, light-driven C1->O1, C2->O2)
      const dc1 = -ka1 * c1 + C_KD1 * o1 + C_KR * c2;
      const do1 = ka1 * c1 - (C_KD1 + C_E12) * o1 + C_E21 * o2;
      const do2 = ka2 * c2 - (C_KD2 + C_E21) * o2 + C_E12 * o1;
      const dc2 = -(ka2 + C_KR) * c2 + C_KD2 * o2;
      c1 += HH_DT * dc1; o1 += HH_DT * do1; o2 += HH_DT * do2; c2 += HH_DT * dc2;
      c1 = Math.max(0, c1); o1 = Math.max(0, o1); o2 = Math.max(0, o2); c2 = Math.max(0, c2);
      const sum = c1 + o1 + o2 + c2 || 1;
      c1 /= sum; o1 /= sum; o2 /= sum; c2 /= sum;

      // ATP-dependent pump strength compresses reversal potentials when ATP collapses
      const pump = Math.min(1, (atp / (0.8 + atp)) / pumpRest);
      const eNa = 50.0 * pump, eK = -77.0 * pump;

      const iNa = 120.0 * m * m * m * h * (v - eNa);
      const iK = 36.0 * n * n * n * n * (v - eK);
      const iL = 0.3 * (v + 54.387);
      iPhoto = -G_CHR2 * (o1 + 0.1 * o2) * (v - 0.0);     // depolarising (positive) for v < 0
      const dv = -iNa - iK - iL + iPhoto + tonic;
      const vNext = Math.max(-100, Math.min(60, v + HH_DT * dv));
      if (v < 0 && vNext >= 0) spiked = true;
      v = vNext;

      const x1 = v + 40, x2 = v + 55;
      const am = Math.abs(x1) < 1e-6 ? 1.0 : (0.1 * x1) / (1 - Math.exp(-x1 / 10));
      const bm = 4 * Math.exp(-(v + 65) / 18);
      const ah = 0.07 * Math.exp(-(v + 65) / 20);
      const bh = 1 / (1 + Math.exp(-(v + 35) / 10));
      const an = Math.abs(x2) < 1e-6 ? 0.1 : (0.01 * x2) / (1 - Math.exp(-x2 / 10));
      const bn = 0.125 * Math.exp(-(v + 65) / 80);
      const sm = phi * (am + bm), sh = phi * (ah + bh), sn = phi * (an + bn);
      m += (am / (am + bm) - m) * (1 - Math.exp(-HH_DT * sm));
      h += (ah / (ah + bh) - h) * (1 - Math.exp(-HH_DT * sh));
      n += (an / (an + bn) - n) * (1 - Math.exp(-HH_DT * sn));

      // Michaelis-Menten ATP production from glucose vs. pump + baseline consumption
      const prod = (0.004 * (glc / (1 + glc))) / 0.8182;
      const cons = 0.004 * (atp / ATP_REST) + 1e-4 * Math.abs(iNa);
      atp = Math.max(0.05, Math.min(4.0, atp + HH_DT * (prod - cons)));
      glc = Math.max(0.1, glc + HH_DT * (-0.02 * prod + 1e-3 * (GLC_REST - glc)));
    }

    if (spiked) spikesThisFrame++;
    view[b + F.VM] = v; view[b + F.M] = m; view[b + F.H] = h; view[b + F.N] = n;
    view[b + F.C1] = c1; view[b + F.O1] = o1; view[b + F.O2] = o2; view[b + F.C2] = c2;
    view[b + F.ATP] = atp; view[b + F.GLC] = glc; view[b + F.ICHR2] = iPhoto;
    view[b + F.SPIKE] = spiked ? 1.0 : 0.0; view[b + F.TEMP] = temp;
  }
  return spikesThisFrame;
}

// ---- Main loop ----
let spikeAccum = 0;
let statsWindowStart = 0;
let lastSnapshotMs = 0;

async function simulationLoop() {
  if (!isRunning || !sharedFloatView) return;
  try {
    const effectiveLaser = throttledByLedger ? simParams.p_laser * 0.25 : simParams.p_laser;
    let frameSpikes = 0;

    if (gpuDevice && computePipeline && neuronBuffer && stagingBuffer && paramsBuffer && bindGroup) {
      try {
        gpuParamsData[0] = simParams.dt;
        gpuParamsData[1] = effectiveLaser;
        gpuParamsData[2] = simParams.na;
        gpuParamsData[3] = simParams.wavelength;
        gpuParamsData[4] = simParams.tau_pulse;
        gpuParamsData[5] = simParams.f_rep;
        gpuParamsData[6] = simParams.l_s;
        gpuParamsData[7] = simParams.market_modulation;
        gpuParamsData[8] = simParams.slm_focal_z;
        gpuParamsData[9] = ORGANOID_RADIUS_UM;
        gpuParamsData[10] = SLM_GRID;
        gpuParamsData[11] = 0.0;
        gpuDevice.queue.writeBuffer(paramsBuffer, 0, gpuParamsData.buffer as ArrayBuffer, gpuParamsData.byteOffset, gpuParamsData.byteLength);
        const enc = gpuDevice.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(computePipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(NEURON_COUNT / 64));
        pass.end();
        enc.copyBufferToBuffer(neuronBuffer, 0, stagingBuffer, 0, sharedFloatView.byteLength);
        gpuDevice.queue.submit([enc.finish()]);
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        sharedFloatView.set(new Float32Array(stagingBuffer.getMappedRange()));
        stagingBuffer.unmap();
        for (let i = 0; i < NEURON_COUNT; i++) frameSpikes += sharedFloatView[i * FLOATS_PER_NEURON + F.SPIKE] > 0.5 ? 1 : 0;
      } catch (err) {
        console.error('GPU step failed, switching to CPU physics:', err);
        gpuDevice = null;
      }
    }
    if (!gpuDevice) frameSpikes = stepCPUPhysics(sharedFloatView, effectiveLaser);

    const now = performance.now();
    spikeAccum += frameSpikes;
    if (now - statsWindowStart >= 250) {
      const sec = (now - statsWindowStart) / 1000;
      self.postMessage({
        type: 'STATS',
        payload: {
          spikesPerNeuronPerSec: spikeAccum / NEURON_COUNT / sec, // wall-clock rate
          gsEfficiency, gsUniformity, backend: gpuDevice ? 'webgpu' : 'cpu',
        },
      });
      spikeAccum = 0;
      statsWindowStart = now;
    }
    if (useSnapshots && now - lastSnapshotMs >= 100) {
      const copy = sharedFloatView.slice();
      self.postMessage({ type: 'SNAPSHOT', buffer: copy.buffer }, [copy.buffer]);
      lastSnapshotMs = now;
    }

    if (gl) renderVolumetricWebGL2(sharedFloatView, now);
    else renderFallback2D(sharedFloatView, now);
  } catch (err) {
    console.error('Simulation frame error:', err);
  }
  setTimeout(simulationLoop, 16); // always reschedule so one bad frame cannot kill the loop
}

self.onmessage = async (event: MessageEvent) => {
  const { type, payload } = event.data;

  if (type === 'INIT') {
    useSnapshots = !(typeof SharedArrayBuffer !== 'undefined' && payload.sharedBuffer instanceof SharedArrayBuffer);
    sharedFloatView = new Float32Array(payload.sharedBuffer);
    offscreenCanvas = payload.canvas;
    let hasWebGL2 = false;
    if (offscreenCanvas) {
      hasWebGL2 = initWebGL2VolumetricPipeline(offscreenCanvas);
      if (!hasWebGL2) ctx2d = offscreenCanvas.getContext('2d');
    }
    seedOrganoidPopulation(sharedFloatView);
    rebuildHologram();
    const hasWebGPU = await initWebGPU(sharedFloatView);
    isRunning = true;
    statsWindowStart = performance.now();
    self.postMessage({
      type: 'READY',
      payload: { hasWebGPU, hasWebGL2, neuronCount: NEURON_COUNT, dendriteEdges: DENDRITE_EDGE_COUNT, snapshots: useSnapshots },
    });
    simulationLoop();
  } else if (type === 'UPDATE_PARAMS') {
    simParams = { ...simParams, ...payload };
    if (payload.market_modulation !== undefined || payload.market_imbalance !== undefined) rebuildHologram();
  } else if (type === 'RESIZE') {
    if (offscreenCanvas) { offscreenCanvas.width = payload.width; offscreenCanvas.height = payload.height; }
  } else if (type === 'THROTTLE_LEDGER') {
    throttledByLedger = Boolean(payload.throttle);
  }
};
