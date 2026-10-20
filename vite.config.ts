import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";

export default defineConfig({
  plugins: [react()],
  // Relative asset paths, so the built `dist/` works from any host or
  // sub-path (Vercel, Netlify, Cloudflare Pages, GitHub Pages).
  base: "./",
  server: { port: 5180 },
});
