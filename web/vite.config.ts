import path from "node:path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// https://vite.dev/config/
// Production Docker/HF: `VITE_BASE=/sim/ npm run build` so assets load under /sim/.
const base = process.env.VITE_BASE?.trim() || "/"

export default defineConfig({
  base,
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  server: {
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
})
