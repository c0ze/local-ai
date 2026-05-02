#!/usr/bin/env node
import { existsSync } from 'node:fs';
import { spawn } from 'node:child_process';

const viteArgs = process.argv.slice(2);
const mcpConfig = process.env.MCP_CONFIG
  || (existsSync('mcp.config.json') ? 'mcp.config.json' : 'mcp.config.example.json');

const children = [];
let shuttingDown = false;

function run(name, command, args) {
  const child = spawn(command, args, {
    stdio: 'inherit',
    shell: process.platform === 'win32',
    env: process.env,
  });

  children.push({ name, child });

  child.on('exit', (code, signal) => {
    if (shuttingDown) return;
    shuttingDown = true;
    const reason = signal ? `${signal}` : `exit ${code ?? 0}`;
    console.error(`[dev] ${name} stopped (${reason}); stopping the rest.`);
    stopAll();
    process.exit(code ?? (signal ? 1 : 0));
  });
}

function stopAll() {
  for (const { child } of children) {
    if (!child.killed) child.kill('SIGTERM');
  }
}

process.on('SIGINT', () => {
  shuttingDown = true;
  stopAll();
});

process.on('SIGTERM', () => {
  shuttingDown = true;
  stopAll();
});

console.log(`[dev] Starting Vite and MCP bridge (${mcpConfig})`);
run('mcp bridge', 'node', ['scripts/mcp-bridge.mjs', mcpConfig]);
run('vite', 'vite', viteArgs);
