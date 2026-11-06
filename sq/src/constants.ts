// Shared constants. Importing these never executes worker code (previously main.ts
// imported them from physics.worker, which ran the worker's top-level code on the UI thread).
export const NEURON_COUNT = 10240;
export const FLOATS_PER_NEURON = 16; // 64 bytes per neuron
export const DENDRITE_EDGE_COUNT = 1800;
export const RESERVOIR_BINS = 64;    // 4 x 4 x 4 spatial grid
export const ORGANOID_RADIUS_UM = 360;
export const SLM_GRID = 32;          // 32 x 32 = 1024 SLM pixels
export const THERMAL_LIMIT_C = 38.5;
export const ATP_FLOOR_MM = 1.0;

// Telemetry layout (float index inside each neuron record)
export const F = {
  X: 0, Y: 1, Z: 2, VM: 3, M: 4, H: 5, N: 6,
  C1: 7, O1: 8, O2: 9, C2: 10,
  ATP: 11, GLC: 12, ICHR2: 13, SPIKE: 14, TEMP: 15,
} as const;

/** Spatial bin (0..63) of a neuron: 4x4x4 grid over the organoid bounding cube. */
export function spatialBin(x: number, y: number, z: number): number {
  const q = (v: number) =>
    Math.max(0, Math.min(3, Math.floor(((v + ORGANOID_RADIUS_UM) / (2 * ORGANOID_RADIUS_UM)) * 4)));
  return q(x) + 4 * q(y) + 16 * q(z);
}
