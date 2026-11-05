import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react-swc";

export default defineConfig({
  plugins: [react()],
  base: "./",
  server: { port: 5181 },
  // ".check.ts", so talk-meter's own `npm test` at the repo root doesn't pick these up.
  test: { include: ["tests/**/*.check.ts"] },
});
