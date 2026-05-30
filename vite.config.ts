import { defineConfig } from "vite";
// WebGPU only runs in modern browsers, so target esnext (enables top-level await).
export default defineConfig({ server: { port: 5173 }, build: { target: "esnext" } });
