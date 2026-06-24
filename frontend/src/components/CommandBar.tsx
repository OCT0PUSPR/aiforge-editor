/**
 * Cmd/Ctrl+K agentic-edit bar. Issues an instruction against the active file
 * (or, in multi-file mode, all open files), calls the workspace AI endpoint,
 * and opens the DiffModal with the proposed change for accept/reject.
 */
import { useEffect, useRef, useState } from "react";
import { ai } from "../api/client";
import { useStore } from "../store";

export function CommandBar() {
  const [open, setOpen] = useState(false);
  const [instruction, setInstruction] = useState("");
  const [multi, setMulti] = useState(false);
  const [busy, setBusy] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const activeTab = useStore((s) => s.activeTab());
  const tabs = useStore((s) => s.tabs);
  const workspace = useStore((s) => s.activeWorkspace);
  const setDiffPreview = useStore((s) => s.setDiffPreview);
  const toast = useStore((s) => s.toast);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && !e.shiftKey && e.key.toLowerCase() === "k") {
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
    if (!workspace) return;
    const text = instruction.trim();
    if (!text) return;
    setBusy(true);
    try {
      if (multi) {
        const paths = tabs.map((t) => t.path);
        if (paths.length === 0) {
          toast("error", "Open files before a multi-file edit");
          setBusy(false);
          return;
        }
        const proposal = await ai(workspace.id).proposeMulti(paths, text);
        setDiffPreview({
          paths,
          diff: proposal.diff,
          files: proposal.files.filter((f) => f.changed),
          original: "",
          instruction: text,
          multifile: true,
        });
        toast(proposal.changed ? "info" : "error", proposal.changed ? "Review the edit" : "No changes proposed");
      } else {
        if (!activeTab) {
          toast("error", "Open a file before running an AI edit");
          setBusy(false);
          return;
        }
        const proposal = await ai(workspace.id).propose(activeTab.path, text);
        setDiffPreview({
          paths: [proposal.path],
          diff: proposal.diff,
          newContent: proposal.new_content,
          original: activeTab.saved,
          instruction: text,
          multifile: false,
        });
        toast(proposal.changed ? "info" : "error", proposal.changed ? "Review the edit" : "No changes proposed");
      }
      setOpen(false);
      setInstruction("");
    } catch (e) {
      toast("error", `Edit failed: ${(e as Error).message}`);
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
            multi
              ? `Instruct an edit across ${tabs.length} open file(s)…`
              : activeTab
                ? `Instruct an edit to ${activeTab.path.split("/").pop()}…`
                : "Open a file first"
          }
          onChange={(e) => setInstruction(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void submit();
          }}
          disabled={busy}
        />
        <label className="cmd-multi" title="Edit all open files">
          <input type="checkbox" checked={multi} onChange={(e) => setMulti(e.target.checked)} />
          multi
        </label>
        <button onClick={() => void submit()} disabled={busy}>
          {busy ? "…" : "Propose"}
        </button>
      </div>
    </div>
  );
}
