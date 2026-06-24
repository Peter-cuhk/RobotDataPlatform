import { copyFile, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const source = resolve(
  root,
  "apps/web/node_modules/@rerun-io/web-viewer/re_viewer_bg.wasm",
);
const destination = resolve(
  root,
  "apps/web/public/rerun/re_viewer_bg.wasm",
);

await mkdir(dirname(destination), { recursive: true });
await copyFile(source, destination);
console.log(`Prepared Rerun WebViewer WASM at ${destination}`);
