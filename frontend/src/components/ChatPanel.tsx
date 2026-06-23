/**
 * Right-sidebar chat panel. Asks the backend about the codebase, streams the
 * answer via SSE, renders code blocks, shows retrieved references, and offers
 * an "apply code to file" action for fenced code blocks.
 */
import { useRef, useState } from "react";
import { streamChat, type ChatTurn, type SearchResult } from "../api/client";
import { useStore } from "../store";

interface Message {
  role: "user" | "assistant";
  content: string;
  references?: SearchResult[];
}

interface Segment {
  type: "text" | "code";
  content: string;
  lang?: string;
}

/** Split assistant text into prose and fenced code blocks. */
function segmentize(text: string): Segment[] {
  const segments: Segment[] = [];
  const re = /```(\w*)\n([\s\S]*?)```/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) segments.push({ type: "text", content: text.slice(last, m.index) });
    segments.push({ type: "code", lang: m[1] || "", content: m[2] });
    last = re.lastIndex;
  }
  if (last < text.length) segments.push({ type: "text", content: text.slice(last) });
  return segments;
}

function AssistantBody({ message }: { message: Message }) {
  const activeTab = useStore((s) => s.activeTab());
  const updateContent = useStore((s) => s.updateContent);
  const setStatus = useStore((s) => s.setStatus);

  const applyToFile = (code: string) => {
    if (!activeTab) {
      setStatus("open a file before applying code");
      return;
    }
    updateContent(activeTab.path, code);
    setStatus(`inserted code into ${activeTab.path} (unsaved)`);
  };

  return (
    <div className="msg-body">
      {segmentize(message.content).map((seg, i) =>
        seg.type === "code" ? (
          <div key={i} className="code-block">
            <div className="code-head">
              <span>{seg.lang || "code"}</span>
              <button onClick={() => applyToFile(seg.content)}>apply to file</button>
            </div>
            <pre>{seg.content}</pre>
          </div>
        ) : (
          <p key={i} className="prose">
            {seg.content}
          </p>
        ),
      )}
      {message.references && message.references.length > 0 && (
        <div className="refs">
          <div className="refs-title">references</div>
          {message.references.map((r, i) => (
            <ReferenceLink key={i} reference={r} />
          ))}
        </div>
      )}
    </div>
  );
}

function ReferenceLink({ reference }: { reference: SearchResult }) {
  const openFile = useStore((s) => s.openFile);
  return (
    <button className="ref-link" onClick={() => openFile(reference.path)}>
      {reference.path}:{reference.start_line}-{reference.end_line}
      {reference.symbol ? ` (${reference.symbol})` : ""}
    </button>
  );
}

export function ChatPanel() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const activeTab = useStore((s) => s.activeTab());
  const abortRef = useRef<AbortController | null>(null);

  const send = async () => {
    const question = input.trim();
    if (!question || streaming) return;
    setInput("");
    const history: ChatTurn[] = messages.map((m) => ({ role: m.role, content: m.content }));
    setMessages((prev) => [
      ...prev,
      { role: "user", content: question },
      { role: "assistant", content: "" },
    ]);
    setStreaming(true);
    const controller = new AbortController();
    abortRef.current = controller;

    await streamChat(
      {
        question,
        open_path: activeTab?.path ?? "",
        open_content: activeTab?.content ?? "",
        history,
        top_k: 6,
      },
      {
        onToken: (text) => {
          setMessages((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            next[next.length - 1] = { ...last, content: last.content + text };
            return next;
          });
        },
        onMeta: (data) => {
          const refs = (data.references as SearchResult[]) ?? [];
          setMessages((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            next[next.length - 1] = { ...last, references: refs };
            return next;
          });
        },
        onError: (msg) => {
          setMessages((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            next[next.length - 1] = { ...last, content: last.content + `\n\n[error: ${msg}]` };
            return next;
          });
        },
        onDone: () => setStreaming(false),
      },
      controller.signal,
    );
    setStreaming(false);
  };

  return (
    <aside className="chat-panel">
      <div className="chat-header">AI CHAT</div>
      <div className="chat-scroll">
        {messages.length === 0 && (
          <div className="chat-empty">
            Ask about your codebase. Answers stream with code references you can
            jump to.
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`msg msg-${m.role}`}>
            <div className="msg-role">{m.role}</div>
            {m.role === "assistant" ? (
              <AssistantBody message={m} />
            ) : (
              <div className="msg-body">{m.content}</div>
            )}
          </div>
        ))}
      </div>
      <div className="chat-input">
        <textarea
          value={input}
          placeholder="Ask about the codebase…"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send();
            }
          }}
        />
        <button disabled={streaming} onClick={() => void send()}>
          {streaming ? "…" : "Send"}
        </button>
      </div>
    </aside>
  );
}
