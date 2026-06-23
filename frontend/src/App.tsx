/**
 * Top-level layout: VS-Code-like shell with a file-tree sidebar, the editor in
 * the centre, an AI chat panel on the right, a command bar (Cmd/Ctrl+K), a diff
 * preview modal, and a status bar.
 */
import { useEffect } from "react";
import { FileTree } from "./components/FileTree";
import { EditorPane } from "./components/EditorPane";
import { ChatPanel } from "./components/ChatPanel";
import { CommandBar } from "./components/CommandBar";
import { DiffModal } from "./components/DiffModal";
import { StatusBar } from "./components/StatusBar";
import { useStore } from "./store";

export default function App() {
  const buildIndex = useStore((s) => s.buildIndex);

  // Build the RAG index once on startup so chat has context immediately.
  useEffect(() => {
    void buildIndex();
  }, [buildIndex]);

  return (
    <div className="app">
      <header className="titlebar">
        <span className="logo">✦ aiforge</span>
        <span className="subtitle">AI-native code editor</span>
      </header>
      <div className="main">
        <FileTree />
        <EditorPane />
        <ChatPanel />
      </div>
      <StatusBar />
      <CommandBar />
      <DiffModal />
    </div>
  );
}
