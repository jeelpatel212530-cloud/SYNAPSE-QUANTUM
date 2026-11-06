// Self-host ONNX Runtime WASM binaries so they load under COOP/COEP (require-corp).
import { cpSync, existsSync, mkdirSync, readdirSync } from 'node:fs';
const src = 'node_modules/onnxruntime-web/dist';
const dst = 'public/ort';
if (existsSync(src)) {
  mkdirSync(dst, { recursive: true });
  for (const f of readdirSync(src)) if (f.endsWith('.wasm') || f.endsWith('.mjs')) cpSync(`${src}/${f}`, `${dst}/${f}`);
  console.log('[copy-ort] WASM runtime copied to public/ort');
}
