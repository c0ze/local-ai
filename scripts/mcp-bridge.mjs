#!/usr/bin/env node
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { SSEClientTransport } from '@modelcontextprotocol/sdk/client/sse.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';

const configPath = resolve(process.argv[2] ?? 'mcp.config.json');
const clients = new Map();

async function loadConfig() {
  try {
    const raw = await readFile(configPath, 'utf8');
    const config = JSON.parse(raw);
    return {
      port: Number(process.env.MCP_BRIDGE_PORT ?? config.port ?? 8787),
      servers: config.servers ?? {},
    };
  } catch (error) {
    if (error?.code !== 'ENOENT') throw error;
    return {
      port: Number(process.env.MCP_BRIDGE_PORT ?? 8787),
      servers: {},
    };
  }
}

const config = await loadConfig();

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type,Accept,Access-Control-Request-Private-Network',
  'Access-Control-Allow-Private-Network': 'true',
};

function json(res, status, data) {
  const body = JSON.stringify(data);
  res.writeHead(status, {
    ...corsHeaders,
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(body),
  });
  res.end(body);
}

async function readBody(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  const raw = Buffer.concat(chunks).toString('utf8');
  return raw ? JSON.parse(raw) : {};
}

function getServerConfig(name) {
  const server = config.servers[name];
  if (!server) throw new Error(`Unknown MCP server "${name}"`);
  return server;
}

function normalizeServerUrls(values) {
  const urls = Array.isArray(values) ? values : [];
  const normalized = [];
  for (const value of urls) {
    if (typeof value !== 'string') continue;
    try {
      const url = new URL(value.trim());
      if (url.protocol !== 'http:' && url.protocol !== 'https:') continue;
      const clean = `${url.origin}${url.pathname === '/' ? '' : url.pathname}`;
      if (!normalized.includes(clean)) normalized.push(clean);
    } catch {
      // Ignore malformed user-provided URLs.
    }
  }
  return normalized;
}

function dynamicServersFromUrls(urls) {
  return Object.fromEntries(normalizeServerUrls(urls).map((url, index) => [
    `mcp_${index + 1}`,
    { transport: 'http', url },
  ]));
}

function allServers(dynamicServers = {}) {
  return {
    ...config.servers,
    ...dynamicServers,
  };
}

async function connectWithHttpFallback(name, url) {
  try {
    const client = new Client({ name: `local-ai-chat-${name}`, version: '1.0.0' });
    const transport = new StreamableHTTPClientTransport(new URL(url));
    await client.connect(transport);
    return { client, transport };
  } catch {
    const client = new Client({ name: `local-ai-chat-${name}`, version: '1.0.0' });
    const transport = new SSEClientTransport(new URL(url));
    await client.connect(transport);
    return { client, transport };
  }
}

async function connectServer(name, dynamicServers = {}) {
  const servers = allServers(dynamicServers);
  const server = servers[name];
  if (!server) throw new Error(`Unknown MCP server "${name}"`);

  const clientKey = server.url ? `${name}:${server.url}` : name;
  if (clients.has(clientKey)) return clients.get(clientKey).client;
  let connection;

  if (server.transport === 'http' || server.transport === 'streamable-http' || server.url) {
    if (!server.url) throw new Error(`MCP server "${name}" is missing url`);
    connection = await connectWithHttpFallback(name, server.url);
  } else {
    if (!server.command) throw new Error(`MCP server "${name}" is missing command`);
    const transport = new StdioClientTransport({
      command: server.command,
      args: Array.isArray(server.args) ? server.args : [],
      env: { ...process.env, ...(server.env ?? {}) },
    });
    const client = new Client({ name: `local-ai-chat-${name}`, version: '1.0.0' });
    await client.connect(transport);
    connection = { client, transport };
  }

  clients.set(clientKey, connection);
  return connection.client;
}

async function listTools(dynamicServers = {}) {
  const tools = [];
  const errors = [];
  const servers = allServers(dynamicServers);

  for (const name of Object.keys(servers)) {
    try {
      const client = await connectServer(name, dynamicServers);
      const result = await client.listTools();
      for (const tool of result.tools ?? []) {
        tools.push({
          server: name,
          name: tool.name,
          description: tool.description,
          inputSchema: tool.inputSchema,
        });
      }
    } catch (error) {
      errors.push({ server: name, error: error instanceof Error ? error.message : String(error) });
    }
  }

  return { tools, errors };
}

async function callTool(body) {
  const server = typeof body.server === 'string' ? body.server : '';
  const name = typeof body.name === 'string' ? body.name : '';
  if (!server || !name) throw new Error('Expected "server" and "name" in request body');
  const dynamicServers = dynamicServersFromUrls(body.serverUrls);

  const client = await connectServer(server, dynamicServers);
  return client.callTool({
    name,
    arguments: body.arguments && typeof body.arguments === 'object' ? body.arguments : {},
  });
}

const httpServer = createServer(async (req, res) => {
  try {
    if (req.method === 'OPTIONS') {
      res.writeHead(204, corsHeaders);
      res.end();
      return;
    }

    const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`);
    if (req.method === 'GET' && url.pathname === '/health') {
      const dynamicServers = dynamicServersFromUrls(url.searchParams.getAll('url'));
      json(res, 200, {
        ok: true,
        servers: Object.keys(allServers(dynamicServers)),
      });
      return;
    }

    if (req.method === 'GET' && url.pathname === '/tools') {
      json(res, 200, await listTools(dynamicServersFromUrls(url.searchParams.getAll('url'))));
      return;
    }

    if (req.method === 'POST' && url.pathname === '/tools/call') {
      json(res, 200, await callTool(await readBody(req)));
      return;
    }

    json(res, 404, { error: 'Not found' });
  } catch (error) {
    json(res, 500, { error: error instanceof Error ? error.message : String(error) });
  }
});

async function shutdown() {
  await Promise.all([...clients.values()].map(async ({ client }) => {
    try {
      await client.close();
    } catch {
      // Best-effort shutdown.
    }
  }));
  httpServer.close(() => process.exit(0));
}

process.on('SIGINT', () => void shutdown());
process.on('SIGTERM', () => void shutdown());

httpServer.listen(config.port, () => {
  console.log(`MCP bridge listening on http://localhost:${config.port}`);
  console.log(`Config: ${configPath}`);
});
