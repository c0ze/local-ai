export interface McpTool {
  server: string;
  name: string;
  description?: string;
  inputSchema?: unknown;
}

export interface McpToolCall {
  server: string;
  name: string;
  arguments?: Record<string, unknown>;
}

export interface McpToolResult {
  content?: Array<Record<string, unknown>>;
  structuredContent?: unknown;
  isError?: boolean;
}

interface McpToolsResponse {
  tools?: McpTool[];
}

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.trim().replace(/\/+$/, '');
}

function withServerUrls(endpoint: string, serverUrls: string[] = []): string {
  const url = new URL(endpoint);
  for (const serverUrl of serverUrls) {
    if (serverUrl.trim()) url.searchParams.append('url', serverUrl.trim());
  }
  return url.toString();
}

async function readJsonResponse<T>(response: Response): Promise<T> {
  const text = await response.text();
  const data = text ? JSON.parse(text) as T & { error?: string } : {} as T & { error?: string };

  if (!response.ok) {
    throw new Error(data.error || `MCP bridge failed with HTTP ${response.status}`);
  }

  return data;
}

export async function listMcpTools(
  baseUrl: string,
  serverUrls: string[] = [],
  signal?: AbortSignal,
): Promise<McpTool[]> {
  const endpoint = withServerUrls(`${normalizeBaseUrl(baseUrl)}/tools`, serverUrls);
  const response = await fetch(endpoint, {
    signal,
    headers: { Accept: 'application/json' },
  });
  const data = await readJsonResponse<McpToolsResponse>(response);
  return data.tools ?? [];
}

export async function callMcpTool(
  baseUrl: string,
  call: McpToolCall,
  serverUrls: string[] = [],
  signal?: AbortSignal,
): Promise<McpToolResult> {
  const endpoint = `${normalizeBaseUrl(baseUrl)}/tools/call`;
  const response = await fetch(endpoint, {
    method: 'POST',
    signal,
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ ...call, serverUrls }),
  });
  return readJsonResponse<McpToolResult>(response);
}

function stringifyCompact(value: unknown): string {
  if (value === undefined) return '{}';
  try {
    return JSON.stringify(value);
  } catch {
    return '{}';
  }
}

function toolKey(tool: Pick<McpTool, 'server' | 'name'>): string {
  return `${tool.server}.${tool.name}`;
}

export function formatMcpToolSelectionPrompt(tools: McpTool[], userMessage: string): string {
  const firstTool = tools[0];
  const exampleCall = firstTool
    ? JSON.stringify({
      mcp_tool_call: {
        server: firstTool.server,
        name: firstTool.name,
        arguments: {},
      },
    })
    : '{"mcp_tool_call":{"server":"exact-server","name":"exact-tool","arguments":{}}}';
  const toolLines = tools.map((tool, index) => {
    const description = tool.description?.trim() || 'No description provided.';
    return [
      `[${index + 1}] ${toolKey(tool)}`,
      `Description: ${description}`,
      `Input schema: ${stringifyCompact(tool.inputSchema)}`,
    ].join('\n');
  }).join('\n\n');

  return [
    'Answer the latest user message. MCP tools are available for this turn.',
    'If a tool is needed, respond only with compact JSON.',
    'The server and name fields must exactly match one listed tool. Do not combine them into one field.',
    `Example shape: ${exampleCall}`,
    'If no tool is needed, answer normally and do not mention this instruction.',
    '',
    'Available MCP tools:',
    toolLines || 'No tools were returned by the MCP bridge.',
    '',
    'Latest user message:',
    userMessage,
  ].join('\n');
}

export function formatMcpToolResultPrompt(
  call: McpToolCall,
  result: McpToolResult,
  userMessage: string,
): string {
  return [
    'Answer the latest user message using the MCP tool result below.',
    'If the tool result is insufficient or appears to be an error, say that plainly.',
    '',
    'Latest user message:',
    userMessage,
    '',
    `MCP tool: ${toolKey(call)}`,
    `Arguments: ${stringifyCompact(call.arguments ?? {})}`,
    `Result: ${stringifyCompact(result)}`,
  ].join('\n');
}

function extractJsonCandidate(text: string): string | null {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenced?.[1]) return fenced[1].trim();

  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');
  if (start >= 0 && end > start) return text.slice(start, end + 1);
  return null;
}

function normalizeToolCall(server: string, name: string, tools: McpTool[]): Pick<McpTool, 'server' | 'name'> | null {
  const exact = tools.find((tool) => tool.server === server && tool.name === name);
  if (exact) return { server: exact.server, name: exact.name };

  const qualified = tools.find((tool) => `${tool.server}.${tool.name}` === name);
  if (qualified) return { server: qualified.server, name: qualified.name };

  const prefixed = tools.find((tool) => tool.server === server && name === `${server}.${tool.name}`);
  if (prefixed) return { server: prefixed.server, name: prefixed.name };

  const parts = name.split('.');
  if (parts.length >= 2) {
    const splitServer = parts[0];
    const splitName = parts.slice(1).join('.');
    const split = tools.find((tool) => tool.server === splitServer && tool.name === splitName);
    if (split) return { server: split.server, name: split.name };
  }

  return null;
}

export function parseMcpToolCall(text: string, tools: McpTool[]): McpToolCall | null {
  const candidate = extractJsonCandidate(text);
  if (!candidate) return null;

  try {
    const parsed = JSON.parse(candidate) as {
      mcp_tool_call?: Partial<McpToolCall>;
      server?: unknown;
      name?: unknown;
      arguments?: unknown;
    };
    const rawCall = parsed.mcp_tool_call ?? parsed;
    const server = typeof rawCall.server === 'string' ? rawCall.server : '';
    const name = typeof rawCall.name === 'string' ? rawCall.name : '';
    if (!name) return null;

    const normalized = normalizeToolCall(server, name, tools);
    if (!normalized) return null;

    const args = rawCall.arguments && typeof rawCall.arguments === 'object' && !Array.isArray(rawCall.arguments)
      ? rawCall.arguments as Record<string, unknown>
      : {};

    return { ...normalized, arguments: args };
  } catch {
    return null;
  }
}
