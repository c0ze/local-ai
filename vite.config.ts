import { defineConfig } from 'vite';

const base = process.env.VITE_BASE_URL || '/';

export default defineConfig({
  base,
  server: {
    allowedHosts: ['cachyos'],
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    target: 'es2022',
  },
  optimizeDeps: {
    exclude: ['@huggingface/transformers'],
  },
});
