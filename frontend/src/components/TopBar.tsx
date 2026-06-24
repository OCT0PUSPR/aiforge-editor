/** Title bar with workspace switcher, user menu, and settings. */
import { useState } from "react";
import { useStore } from "../store";

export function TopBar() {
  const user = useStore((s) => s.user);
  const workspaces = useStore((s) => s.workspaces);
  const active = useStore((s) => s.activeWorkspace);
  const switchWorkspace = useStore((s) => s.switchWorkspace);
  const createWorkspace = useStore((s) => s.createWorkspace);
  const logout = useStore((s) => s.logout);
  const setSettingsOpen = useStore((s) => s.setSettingsOpen);
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");

  const onCreate = async () => {
    if (name.trim()) {
      await createWorkspace(name.trim());
      setName("");
      setCreating(false);
    }
  };

  return (
    <header className="titlebar">
      <span className="logo">✦ aiforge</span>
      <div className="ws-switcher">
        <select
          value={active?.id ?? ""}
          onChange={(e) => switchWorkspace(e.target.value)}
          aria-label="Workspace"
        >
          {workspaces.map((w) => (
            <option key={w.id} value={w.id}>
              {w.name}
            </option>
          ))}
        </select>
        {creating ? (
          <span className="ws-create">
            <input
              autoFocus
              placeholder="New workspace name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") void onCreate();
                if (e.key === "Escape") setCreating(false);
              }}
            />
            <button onClick={() => void onCreate()}>add</button>
          </span>
        ) : (
          <button className="ws-add" title="New workspace" onClick={() => setCreating(true)}>
            +
          </button>
        )}
      </div>
      <div className="topbar-right">
        <button title="Settings" onClick={() => setSettingsOpen(true)}>
          ⚙
        </button>
        <span className="user-name">{user?.username}</span>
        <button title="Sign out" onClick={logout}>
          ⎋
        </button>
      </div>
    </header>
  );
}
