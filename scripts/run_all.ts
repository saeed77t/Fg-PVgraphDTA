// Optional TypeScript launcher; scientific computation stays in Python.
// Node with TypeScript support: node --experimental-strip-types scripts/run_all.ts --data ...
import { spawnSync } from "node:child_process";

const python: string = process.env.PYTHON_EXECUTABLE ?? "python";
const result = spawnSync(python, ["-m", "fgpvdta", "suite", "--name", "all", ...process.argv.slice(2)], {
  stdio: "inherit",
  shell: false,
});
if (result.error) {
  console.error(result.error.message);
}
process.exit(result.status ?? 1);
