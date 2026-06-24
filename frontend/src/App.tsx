/**
 * Top-level app: gates on auth, then renders the VS-Code-like shell (top bar,
 * file tree, editor, chat, status bar) plus the command bar, command palette,
 * diff modal, settings, and toasts.
 */
import { useEffect } from "react";
import { FileTree } from "./components/FileTree";
import { EditorPane } from "./components/EditorPane";
import { ChatPanel } from "./components/ChatPanel";
import { CommandBar } from "./components/CommandBar";
import { CommandPalette } from "./components/CommandPalette";
import { DiffModal } from "./components/DiffModal";
import { StatusBar } from "./components/StatusBar";
import { TopBar } from "./components/TopBar";
import { SettingsModal } from "./components/Settings";
import { Toasts } from "./components/Toasts";
import { Login } from "./components/Login";
import { useStore } from "./store";

export default function App() {
  const status = useStore((s) => s.status);
  const bootstrap = useStore((s) => s.bootstrap);

  useEffect(() => {
    void bootstrap();
  }, [bootstrap]);

  if (status === "init") {
    return (
      <div className="loading-screen">
        <div className="loading-logo">✦ aiforge</div>
        <div className="spinner" />
      </div>
    );
  }

  if (status === "unauthenticated") {
    return (
      <>
        <Login />
        <Toasts />
      </>
    );
  }

  return (
    <div className="app">
      <TopBar />
      <div className="main">
        <FileTree />
        <EditorPane />
        <ChatPanel />
      </div>
      <StatusBar />
      <CommandBar />
      <CommandPalette />
      <DiffModal />
      <SettingsModal />
      <Toasts />
    </div>
  );
}
