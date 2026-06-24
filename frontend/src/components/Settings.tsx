/** Settings modal: completion toggle, RAG top-k, provider info. */
import { useStore } from "../store";

export function SettingsModal() {
  const open = useStore((s) => s.settingsOpen);
  const setOpen = useStore((s) => s.setSettingsOpen);
  const settings = useStore((s) => s.settings);
  const setSettings = useStore((s) => s.setSettings);
  const indexStats = useStore((s) => s.indexStats);

  if (!open) return null;
  return (
    <div className="modal-overlay" onClick={() => setOpen(false)}>
      <div className="settings-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <strong>Settings</strong>
          <button className="icon-btn" onClick={() => setOpen(false)}>
            ×
          </button>
        </div>
        <div className="settings-body">
          <label className="setting-row">
            <span>Inline AI completion</span>
            <input
              type="checkbox"
              checked={settings.completionEnabled}
              onChange={(e) => setSettings({ completionEnabled: e.target.checked })}
            />
          </label>
          <label className="setting-row">
            <span>RAG results (top-k): {settings.topK}</span>
            <input
              type="range"
              min={1}
              max={20}
              value={settings.topK}
              onChange={(e) => setSettings({ topK: Number(e.target.value) })}
            />
          </label>
          <div className="settings-info">
            <div>
              <strong>Provider / model</strong> is configured on the backend
              (env vars). Completion can be served by our own from-scratch model,
              Anthropic Claude, or HuggingFace; chat &amp; edits default to Claude
              in production and a deterministic mock offline.
            </div>
            {indexStats && (
              <div className="settings-stat">
                Index: {indexStats.files} files, {indexStats.chunks} chunks (
                {indexStats.embedder})
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
