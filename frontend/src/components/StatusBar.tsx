/** Bottom status bar: workspace, index stats, active file/dirty, keybindings. */
import { useStore } from "../store";

export function StatusBar() {
  const statusText = useStore((s) => s.statusText);
  const activeTab = useStore((s) => s.activeTab());
  const isDirty = useStore((s) => s.isDirty);
  const indexStats = useStore((s) => s.indexStats);
  const workspace = useStore((s) => s.activeWorkspace);

  return (
    <footer className="status-bar">
      <span className="status-left">{statusText}</span>
      <span className="status-right">
        {workspace && <span className="status-pill">ws: {workspace.name}</span>}
        {indexStats && (
          <span className="status-pill">
            RAG: {indexStats.files}f / {indexStats.chunks}c ({indexStats.embedder})
          </span>
        )}
        {activeTab && (
          <span className="status-pill">
            {activeTab.path}
            {isDirty(activeTab.path) ? " ●" : ""}
          </span>
        )}
        <span className="status-pill">⌘K edit · ⌘⇧P palette · ⌘S save</span>
      </span>
    </footer>
  );
}
