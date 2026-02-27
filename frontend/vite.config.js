import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'node:fs'
import path from 'node:path'

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 3000,
    https: {
      key: fs.readFileSync(path.resolve(__dirname, 'certs/192.168.1.15+3-key.pem')),
      cert: fs.readFileSync(path.resolve(__dirname, 'certs/192.168.1.15+3.pem')),
    },
    proxy: {
      '/v1': { target: 'http://localhost:8000', changeOrigin: true },
      '/artifacts': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
})
