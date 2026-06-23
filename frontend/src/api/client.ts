/**
 * Typed client for the aiforge backend.
 *
 * Plain `fetch` for JSON endpoints, plus an SSE helper that parses the
 * backend's `event:`/`data:` stream contract:
 *
 *   event: token   data: {"text": "..."}     // incremental output
 *   event: meta    data: {"references": [...]} // chat reference payload
 *   event: done    data: {}                   // stream finished
 *   event: error   data: {"message": "..."}   // backend error
 *
 * The SSE helpers read the response body as a stream and dispatch callbacks, so
 * the UI can render tokens as they arrive.
 */

// -- shared types -----------------------------------------------------------
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

export interface EditProposal {
  path: string;
  diff: string;
  new_content: string;
  changed: boolean;
}

export interface ApplyResult {
  path: string;
  new_content: string;
  reverse_diff: string;
}

export interface IndexStats {
  files: number;
  chunks: number;
  embedder: string;
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

const BASE = ""; // same-origin; Vite proxies /api and /health in dev.

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return (await res.json()) as T;
}

// -- workspace --------------------------------------------------------------
export function getTree(path = ""): Promise<TreeNode> {
  return fetch(`${BASE}/api/tree?path=${encodeURIComponent(path)}`).then((r) =>
    json<TreeNode>(r),
  );
}

export function getFile(path: string): Promise<FileContent> {
  return fetch(`${BASE}/api/file?path=${encodeURIComponent(path)}`).then((r) =>
    json<FileContent>(r),
  );
}

export function saveFile(path: string, content: string): Promise<{ saved: boolean }> {
  return fetch(`${BASE}/api/file`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, content }),
  }).then((r) => json(r));
}

export function createFile(path: string, content = ""): Promise<{ created: boolean }> {
  return fetch(`${BASE}/api/file`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, content }),
  }).then((r) => json(r));
}

export function deleteFile(path: string): Promise<{ deleted: boolean }> {
  return fetch(`${BASE}/api/file?path=${encodeURIComponent(path)}`, {
    method: "DELETE",
  }).then((r) => json(r));
}

// -- RAG --------------------------------------------------------------------
export function buildIndex(): Promise<IndexStats> {
  return fetch(`${BASE}/api/index`, { method: "POST" }).then((r) => json<IndexStats>(r));
}

export function search(q: string, k = 6): Promise<{ query: string; results: SearchResult[] }> {
  return fetch(`${BASE}/api/search?q=${encodeURIComponent(q)}&k=${k}`).then((r) => json(r));
}

// -- agentic edit -----------------------------------------------------------
export function proposeEdit(path: string, instruction: string): Promise<EditProposal> {
  return fetch(`${BASE}/api/edit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, instruction }),
  }).then((r) => json<EditProposal>(r));
}

export function applyEdit(args: {
  path: string;
  diff?: string;
  new_content?: string;
  expected_original?: string;
}): Promise<ApplyResult> {
  return fetch(`${BASE}/api/edit/apply`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(args),
  }).then((r) => json<ApplyResult>(r));
}

// -- SSE core ---------------------------------------------------------------
export interface SSEHandlers {
  onToken?: (text: string) => void;
  onMeta?: (data: Record<string, unknown>) => void;
  onError?: (message: string) => void;
  onDone?: () => void;
}

interface SSEEvent {
  event: string;
  data: string;
}

function parseSSEChunk(buffer: string): { events: SSEEvent[]; rest: string } {
  // SSE events are separated by a blank line.
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

async function streamSSE(
  url: string,
  body: unknown,
  handlers: SSEHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${BASE}${url}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    const msg = `${res.status}: ${res.statusText}`;
    handlers.onError?.(msg);
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
      for (const ev of events) {
        dispatch(ev, handlers);
      }
    }
  } catch (err) {
    if ((err as Error)?.name !== "AbortError") {
      handlers.onError?.(String((err as Error)?.message ?? err));
    }
  }
}

function dispatch(ev: SSEEvent, handlers: SSEHandlers): void {
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
    case "error":
      handlers.onError?.(String(payload.message ?? "stream error"));
      break;
    case "done":
      handlers.onDone?.();
      break;
  }
}

// Exposed for unit testing the SSE parser.
export const _internals = { parseSSEChunk, dispatch };

// -- AI streaming endpoints -------------------------------------------------
export function streamComplete(
  args: { prefix: string; suffix?: string; language?: string; path?: string; max_tokens?: number },
  handlers: SSEHandlers,
  signal?: AbortSignal,
): Promise<void> {
  return streamSSE("/api/complete", args, handlers, signal);
}

export function streamChat(
  args: {
    question: string;
    open_path?: string;
    open_content?: string;
    history?: ChatTurn[];
    top_k?: number;
  },
  handlers: SSEHandlers,
  signal?: AbortSignal,
): Promise<void> {
  return streamSSE("/api/chat", args, handlers, signal);
}

/** Convenience: run an inline completion and resolve with the full text. */
export async function completeOnce(
  args: { prefix: string; suffix?: string; language?: string; path?: string; max_tokens?: number },
  signal?: AbortSignal,
): Promise<string> {
  let out = "";
  await streamComplete(
    args,
    {
      onToken: (t) => {
        out += t;
      },
    },
    signal,
  );
  return out;
}
