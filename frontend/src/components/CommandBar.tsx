/**
 * Cmd/Ctrl+K command bar for issuing an agentic edit instruction against the
 * active file. On submit it calls `/api/edit`, then opens the DiffModal with the
 * proposed change for accept/reject.
 */
import { useEffect, useRef, useState } from "react";
import { proposeEdit } from "../api/client";
import { useStore } from "../store";

export function CommandBar() {
  const [open, setOpen] = useState(false);
  const [instruction, setInstruction] = useState("");
  const [busy, setBusy] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const activeTab = useStore((s) => s.activeTab());
  const setDiffPreview = useStore((s) => s.setDiffPreview);
  const setStatus = useStore((s) => s.setStatus);

  // Global Cmd/Ctrl+K toggles the bar.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((o) => !o);
      } else if (e.key === "Escape") {
        setOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  const submit = async () => {
    if (!activeTab) {
      setStatus("open a file before running an AI edit");
      setOpen(false);
      return;
    }
    const text = instruction.trim();
    if (!text) return;
    setBusy(true);
    setStatus(`proposing edit: ${text}`);
    try {
      const proposal = await proposeEdit(activeTab.path, text);
      setDiffPreview({
        path: proposal.path,
        diff: proposal.diff,
        newContent: proposal.new_content,
        original: activeTab.saved,
        instruction: text,
      });
      setStatus(proposal.changed ? "review the proposed edit" : "no changes proposed");
      setOpen(false);
      setInstruction("");
    } catch (e) {
      setStatus(`error: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  if (!open) return null;

  return (
    <div className="cmd-overlay" onClick={() => setOpen(false)}>
      <div className="cmd-bar" onClick={(e) => e.stopPropagation()}>
        <span className="cmd-prefix">✦ edit</span>
        <input
          ref={inputRef}
          value={instruction}
          placeholder={
            activeTab ? `Instruct an edit to ${activeTab.path.split("/").pop()}…` : "Open a file first"
          }
          onChange={(e) => setInstruction(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void submit();
          }}
          disabled={busy}
        />
        <button onClick={() => void submit()} disabled={busy}>
          {busy ? "…" : "Propose"}
        </button>
      </div>
    </div>
  );
}
