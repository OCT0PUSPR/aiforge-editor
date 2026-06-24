/** Command palette (Cmd/Ctrl+Shift+P): fuzzy-filterable actions. */
import { useEffect, useMemo, useRef, useState } from "react";
import { useStore } from "../store";

interface Command {
  id: string;
  label: string;
  run: () => void;
}

export function CommandPalette() {
  const open = useStore((s) => s.paletteOpen);
  const setOpen = useStore((s) => s.setPaletteOpen);
  const saveActive = useStore((s) => s.saveActive);
  const buildIndex = useStore((s) => s.buildIndex);
  const loadTree = useStore((s) => s.loadTree);
  const setSettingsOpen = useStore((s) => s.setSettingsOpen);
  const logout = useStore((s) => s.logout);
  const createFile = useStore((s) => s.createFile);
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key.toLowerCase() === "p") {
        e.preventDefault();
        setOpen(!open);
      } else if (e.key === "Escape" && open) {
        setOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, setOpen]);

  useEffect(() => {
    if (open) {
      setQuery("");
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [open]);

  const commands: Command[] = useMemo(
    () => [
      { id: "save", label: "File: Save active file", run: () => void saveActive() },
      {
        id: "new",
        label: "File: New file…",
        run: () => {
          const path = window.prompt("New file path (e.g. src/util.py)");
          if (path) void createFile(path);
        },
      },
      { id: "reindex", label: "RAG: Rebuild index", run: () => void buildIndex() },
      { id: "refresh", label: "Files: Refresh tree", run: () => void loadTree() },
      { id: "settings", label: "Open settings", run: () => setSettingsOpen(true) },
      { id: "logout", label: "Sign out", run: () => logout() },
    ],
    [saveActive, buildIndex, loadTree, setSettingsOpen, logout, createFile],
  );

  const filtered = commands.filter((c) =>
    c.label.toLowerCase().includes(query.toLowerCase()),
  );

  if (!open) return null;
  return (
    <div className="cmd-overlay" onClick={() => setOpen(false)}>
      <div className="palette" onClick={(e) => e.stopPropagation()}>
        <input
          ref={inputRef}
          className="palette-input"
          placeholder="Type a command…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && filtered[0]) {
              filtered[0].run();
              setOpen(false);
            }
          }}
        />
        <div className="palette-list">
          {filtered.map((c) => (
            <div
              key={c.id}
              className="palette-item"
              onClick={() => {
                c.run();
                setOpen(false);
              }}
            >
              {c.label}
            </div>
          ))}
          {filtered.length === 0 && <div className="palette-empty">no matching commands</div>}
        </div>
      </div>
    </div>
  );
}
