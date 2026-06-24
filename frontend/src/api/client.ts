/**
 * Typed client for the aiforge backend (multi-tenant, authenticated).
 *
 * - JWT access/refresh stored in localStorage; the access token is attached as
 *   a Bearer header on every request, with a transparent refresh-on-401 retry.
 * - All workspace endpoints are scoped under /api/workspaces/{id}/...
 * - SSE helper parses the backend's event/data stream:
 *     event: token   data: {"text": "..."}
 *     event: meta    data: {"references": [...]}
 *     event: heartbeat data: {"t": 123}
 *     event: done    data: {}
 *     event: error   data: {"message": "..."}
 */

// -- types ------------------------------------------------------------------
export interface TreeNode {
  name: string;
  path: string;
  type: "file" | "dir";
  children?: TreeNode[];
}
export interface FileContent {
  path: string;
  content: string;
}
export interface SearchResult {
  path: string;
  start_line: number;
  end_line: number;
  symbol: string | null;
  text: string;
  score: number;
}
export interface Usage {
  provider: string;
  model: string;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
}
export interface EditProposal {
  path: string;
  diff: string;
  new_content: string;
  changed: boolean;
  instruction?: string;
  usage?: Usage | null;
}
export interface FileChange {
  path: string;
  new_content: string;
  changed: boolean;
}
export interface MultiEditProposal {
  files: FileChange[];
  diff: string;
  changed: boolean;
  instruction?: string;
  usage?: Usage | null;
}
export interface IndexStats {
  files: number;
  chunks: number;
  embedder: string;
  added?: number;
  updated?: number;
  removed?: number;
  unchanged?: number;
}
export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}
export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  expires_in: number;
}
export interface User {
  id: string;
  email: string;
  username: string;
  is_admin: boolean;
}
export interface Workspace {
  id: string;
  name: string;
  slug: string;
}
export interface WorkspaceUsage {
  files: number;
  bytes: number;
  max_files: number;
  max_bytes: number;
}
export interface EditHistoryItem {
  id: string;
  path: string;
  instruction: string;
  applied: boolean;
  created_at: string;
}

const BASE = "";
const ACCESS_KEY = "aiforge.access";
const REFRESH_KEY = "aiforge.refresh";

// -- token storage ----------------------------------------------------------
export const tokens = {
  get access(): string | null {
    return localStorage.getItem(ACCESS_KEY);
  },
  get refresh(): string | null {
    return localStorage.getItem(REFRESH_KEY);
  },
  set(t: AuthTokens) {
    localStorage.setItem(ACCESS_KEY, t.access_token);
    localStorage.setItem(REFRESH_KEY, t.refresh_token);
  },
  clear() {
    localStorage.removeItem(ACCESS_KEY);
    localStorage.removeItem(REFRESH_KEY);
  },
};

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

function authHeaders(extra?: HeadersInit): HeadersInit {
  const h: Record<string, string> = { ...(extra as Record<string, string>) };
  const access = tokens.access;
  if (access) h["Authorization"] = `Bearer ${access}`;
  return h;
}

async function refreshAccess(): Promise<boolean> {
  const refresh = tokens.refresh;
  if (!refresh) return false;
  const res = await fetch(`${BASE}/api/auth/refresh`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refresh_token: refresh }),
  });
  if (!res.ok) {
    tokens.clear();
    return false;
  }
  tokens.set((await res.json()) as AuthTokens);
  return true;
}

async function request<T>(path: string, init: RequestInit = {}, retry = true): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: authHeaders({
      "Content-Type": "application/json",
      ...(init.headers as Record<string, string>),
    }),
  });
  if (res.status === 401 && retry && tokens.refresh) {
    if (await refreshAccess()) return request<T>(path, init, false);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* non-JSON */
    }
    throw new ApiError(res.status, `${res.status}: ${detail}`);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

// -- auth -------------------------------------------------------------------
export const auth = {
  async register(email: string, username: string, password: string): Promise<AuthTokens> {
    const t = await request<AuthTokens>("/api/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, username, password }),
    });
    tokens.set(t);
    return t;
  },
  async login(username: string, password: string): Promise<AuthTokens> {
    const t = await request<AuthTokens>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
    tokens.set(t);
    return t;
  },
  me(): Promise<User> {
    return request<User>("/api/auth/me");
  },
  logout() {
    tokens.clear();
  },
};

// -- workspaces -------------------------------------------------------------
export const workspaces = {
  list: () => request<Workspace[]>("/api/workspaces"),
  create: (name: string) =>
    request<Workspace>("/api/workspaces", { method: "POST", body: JSON.stringify({ name }) }),
  remove: (id: string) => request<void>(`/api/workspaces/${id}`, { method: "DELETE" }),
  usage: (id: string) => request<WorkspaceUsage>(`/api/workspaces/${id}/usage`),
};

// -- files ------------------------------------------------------------------
export function files(wsId: string) {
  const base = `/api/workspaces/${wsId}/files`;
  return {
    tree: (path = "") => request<TreeNode>(`${base}/tree?path=${encodeURIComponent(path)}`),
    read: (path: string) => request<FileContent>(`${base}?path=${encodeURIComponent(path)}`),
    save: (path: string, content: string) =>
      request<{ saved: boolean }>(base, { method: "PUT", body: JSON.stringify({ path, content }) }),
    create: (path: string, content = "") =>
      request<{ created: boolean }>(base, { method: "POST", body: JSON.stringify({ path, content }) }),
    rename: (src: string, dst: string) =>
      request<{ renamed: boolean }>(`${base}/rename`, { method: "POST", body: JSON.stringify({ src, dst }) }),
    remove: (path: string) =>
      request<{ deleted: boolean }>(`${base}?path=${encodeURIComponent(path)}`, { method: "DELETE" }),
  };
}

// -- rag --------------------------------------------------------------------
export function rag(wsId: string) {
  const base = `/api/workspaces/${wsId}/rag`;
  return {
    index: () => request<IndexStats>(`${base}/index`, { method: "POST" }),
    reindex: () => request<IndexStats>(`${base}/reindex`, { method: "POST" }),
    search: (q: string, k = 6) =>
      request<{ query: string; results: SearchResult[] }>(`${base}/search?q=${encodeURIComponent(q)}&k=${k}`),
  };
}

// -- ai edits ---------------------------------------------------------------
export function ai(wsId: string) {
  const base = `/api/workspaces/${wsId}/ai`;
  return {
    propose: (path: string, instruction: string) =>
      request<EditProposal>(`${base}/edit`, { method: "POST", body: JSON.stringify({ path, instruction }) }),
    proposeMulti: (paths: string[], instruction: string) =>
      request<MultiEditProposal>(`${base}/edit/multi`, {
        method: "POST",
        body: JSON.stringify({ paths, instruction }),
      }),
    apply: (args: {
      path?: string;
      diff?: string;
      new_content?: string;
      expected_original?: string;
      multifile?: boolean;
      instruction?: string;
    }) => request<Record<string, unknown>>(`${base}/edit/apply`, { method: "POST", body: JSON.stringify(args) }),
    history: () => request<{ history: EditHistoryItem[] }>(`${base}/edit/history`),
    undo: (id: string) => request<{ undone: boolean }>(`${base}/edit/undo/${id}`, { method: "POST" }),
  };
}

// -- SSE --------------------------------------------------------------------
export interface SSEHandlers {
  onToken?: (text: string) => void;
  onMeta?: (data: Record<string, unknown>) => void;
  onError?: (message: string) => void;
  onDone?: () => void;
  onHeartbeat?: () => void;
}
interface SSEEvent {
  event: string;
  data: string;
}

export function parseSSEChunk(buffer: string): { events: SSEEvent[]; rest: string } {
  const events: SSEEvent[] = [];
  const parts = buffer.split("\n\n");
  const rest = parts.pop() ?? "";
  for (const part of parts) {
    let event = "message";
    const dataLines: string[] = [];
    for (const line of part.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    }
    if (dataLines.length) events.push({ event, data: dataLines.join("\n") });
  }
  return { events, rest };
}

export function dispatchSSE(ev: SSEEvent, handlers: SSEHandlers): void {
  let payload: Record<string, unknown> = {};
  try {
    payload = JSON.parse(ev.data);
  } catch {
    payload = {};
  }
  switch (ev.event) {
    case "token":
      handlers.onToken?.(String(payload.text ?? ""));
      break;
    case "meta":
      handlers.onMeta?.(payload);
      break;
    case "heartbeat":
      handlers.onHeartbeat?.();
      break;
    case "error":
      handlers.onError?.(String(payload.message ?? "stream error"));
      break;
    case "done":
      handlers.onDone?.();
      break;
  }
}

async function streamSSE(
  url: string,
  body: unknown,
  handlers: SSEHandlers,
  signal?: AbortSignal,
  retry = true,
): Promise<void> {
  const res = await fetch(`${BASE}${url}`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json", Accept: "text/event-stream" }),
    body: JSON.stringify(body),
    signal,
  });
  if (res.status === 401 && retry && tokens.refresh) {
    if (await refreshAccess()) return streamSSE(url, body, handlers, signal, false);
  }
  if (!res.ok || !res.body) {
    handlers.onError?.(`${res.status}: ${res.statusText}`);
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const { events, rest } = parseSSEChunk(buffer);
      buffer = rest;
      for (const ev of events) dispatchSSE(ev, handlers);
    }
  } catch (err) {
    if ((err as Error)?.name !== "AbortError") {
      handlers.onError?.(String((err as Error)?.message ?? err));
    }
  }
}

export function streamComplete(
  wsId: string,
  args: { prefix: string; suffix?: string; language?: string; path?: string; max_tokens?: number },
  handlers: SSEHandlers,
  signal?: AbortSignal,
): Promise<void> {
  return streamSSE(`/api/workspaces/${wsId}/ai/complete`, args, handlers, signal);
}

export function streamChat(
  wsId: string,
  args: {
    question: string;
    open_path?: string;
    open_content?: string;
    history?: ChatTurn[];
    top_k?: number;
    session_id?: string;
  },
  handlers: SSEHandlers,
  signal?: AbortSignal,
): Promise<void> {
  return streamSSE(`/api/workspaces/${wsId}/ai/chat`, args, handlers, signal);
}

export async function completeOnce(
  wsId: string,
  args: { prefix: string; suffix?: string; language?: string; path?: string; max_tokens?: number },
  signal?: AbortSignal,
): Promise<string> {
  let out = "";
  await streamComplete(wsId, args, { onToken: (t) => (out += t) }, signal);
  return out;
}

export const _internals = { parseSSEChunk, dispatchSSE };
