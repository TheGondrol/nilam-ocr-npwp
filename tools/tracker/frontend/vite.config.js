import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

// /api dan /v1 diteruskan ke backend tracker (tools/tracker/backend/app.py, port 8090),
// jadi UI dan SSE-nya satu origin: tidak perlu CORS.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(import.meta.dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8090', changeOrigin: true },
      '/v1': { target: 'http://127.0.0.1:8090', changeOrigin: true },
    },
  },
})

