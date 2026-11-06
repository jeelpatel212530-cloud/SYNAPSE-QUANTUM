// ============================================================================
// SYNAPSE-QUANTUM: Offline Document Suite (Nodes 1, 3, 5, 12).
// Fixes: real structural redaction (sensitive text is never written), SHA-256 of the telemetry
// record, a SIMULATED watermark, and seeded jitter so output is reproducible.
// ============================================================================
import { PDFDocument, StandardFonts, rgb, degrees } from 'pdf-lib';
import { mulberry32 } from '../util/rng';

export interface TelemetrySummary {
  meanVoltage: string;
  firingRateHz: string;
  impliedVol: string;
  atpLevel: string;
  tempCelsius: string;
  laserMw?: string;
  focalZ?: string;
  alphaBps?: string;
  throttled?: string;
}

export async function sha256Hex(text: string): Promise<string> {
  const buf = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return Array.from(new Uint8Array(buf)).map((b) => b.toString(16).padStart(2, '0')).join('');
}

export async function generateOfflineComplianceBundle(summary: TelemetrySummary): Promise<Uint8Array> {
  const record = JSON.stringify({ ...summary, generatedAt: new Date().toISOString() });
  const digest = await sha256Hex(record);
  const rnd = mulberry32(parseInt(digest.slice(0, 8), 16)); // same record => same "handwriting"

  const pdfDoc = await PDFDocument.create();
  const fontMono = await pdfDoc.embedFont(StandardFonts.Courier);
  const fontBold = await pdfDoc.embedFont(StandardFonts.HelveticaBold);
  const page = pdfDoc.addPage([595.28, 841.89]); // A4
  const { height } = page.getSize();

  page.drawText('SYNAPSE-QUANTUM: WETWARE SOP & TELEMETRY ELN (SIMULATION)', {
    x: 45, y: height - 55, size: 13, font: fontBold, color: rgb(0.05, 0.15, 0.3),
  });

  // Node 1: structural redaction. The identifier is never written, so nothing can be extracted
  // from under the box (drawing a rectangle over real text does not redact it).
  const label = 'CELL LINE DONOR GENOMIC ID: ';
  page.drawText(label, { x: 45, y: height - 85, size: 10, font: fontMono, color: rgb(0.1, 0.1, 0.1) });
  page.drawRectangle({ x: 45 + fontMono.widthOfTextAtSize(label, 10), y: height - 88, width: 175, height: 14, color: rgb(0, 0, 0) });

  const lines = [
    `1. Mean Organoid Potential (Vm) : ${summary.meanVoltage} mV`,
    `2. Spiking rate (per neuron)    : ${summary.firingRateHz} Hz (wall-clock)`,
    `3. Heston-filter Implied Vol    : ${summary.impliedVol}`,
    `4. Intracellular ATP Pool       : ${summary.atpLevel} mM`,
    `5. Max Parenchyma Temperature   : ${summary.tempCelsius} C`,
  ];
  if (summary.laserMw) lines.push(`6. Laser power / focal plane    : ${summary.laserMw} mW / ${summary.focalZ ?? '0'} um`);
  if (summary.alphaBps) lines.push(`7. Readout alpha (signed)       : ${summary.alphaBps} bps  throttled=${summary.throttled ?? 'n/a'}`);

  // Node 5: seeded, reproducible baseline jitter
  lines.forEach((line, idx) => {
    page.drawText(line, {
      x: 50 + (rnd() - 0.5) * 1.4,
      y: height - 135 - idx * 24 + (rnd() - 0.5) * 1.8,
      size: 11, font: fontMono, color: rgb(0.08, 0.22, 0.38),
    });
  });

  // Integrity block
  page.drawText('RECORD SHA-256 (over summary JSON incl. timestamp):', { x: 45, y: 120, size: 8, font: fontBold });
  page.drawText(digest.slice(0, 32), { x: 45, y: 108, size: 8, font: fontMono });
  page.drawText(digest.slice(32), { x: 45, y: 97, size: 8, font: fontMono });
  page.drawText('Donor genomic identifiers are never written to this document (structural redaction).', {
    x: 45, y: 80, size: 8, font: fontMono, color: rgb(0.3, 0.3, 0.3),
  });

  // Watermark: this is simulated data and must not pass as a real lab record
  page.drawText('SIMULATED DATA', {
    x: 110, y: 300, size: 58, font: fontBold, color: rgb(0.85, 0.2, 0.2), opacity: 0.18, rotate: degrees(35),
  });

  return await pdfDoc.save();
}
