/** Bottom status bar: workspace status, index stats, and active file/dirty state. */
import { useStore } from "../store";

export function StatusBar() {
  const status = useStore((s) => s.status);
  const activeTab = useStore((s) => s.activeTab());
  const isDirty = useStore((s) => s.isDirty);
  const indexStats = useStore((s) => s.indexStats);

  return (
    <footer className="status-bar">
      <span className="status-left">{status}</span>
      <span className="status-right">
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
        <span className="status-pill">⌘K edit · ⌘S save</span>
      </span>
    </footer>
  );
}
