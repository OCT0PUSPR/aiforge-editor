/**
 * Monaco-based editor pane with tabs, save-on-Ctrl/Cmd+S, and a real inline AI
 * completion provider that calls `/api/complete` with the prefix/suffix around
 * the cursor (fill-in-the-middle).
 */
import Editor, { type Monaco, type OnMount } from "@monaco-editor/react";
import type { editor, IDisposable } from "monaco-editor";
import { useCallback, useRef } from "react";
import { useStore } from "../store";
import { languageForPath } from "../language";
import { completeOnce } from "../api/client";

function Tabs() {
  const tabs = useStore((s) => s.tabs);
  const activePath = useStore((s) => s.activePath);
  const setActive = useStore((s) => s.setActive);
  const closeTab = useStore((s) => s.closeTab);
  const isDirty = useStore((s) => s.isDirty);
  return (
    <div className="tabs">
      {tabs.map((tab) => (
        <div
          key={tab.path}
          className={`tab${tab.path === activePath ? " active" : ""}`}
          onClick={() => setActive(tab.path)}
          title={tab.path}
        >
          <span className="tab-name">{tab.path.split("/").pop()}</span>
          {isDirty(tab.path) && <span className="tab-dirty">●</span>}
          <span
            className="tab-close"
            onClick={(e) => {
              e.stopPropagation();
              closeTab(tab.path);
            }}
          >
            ×
          </span>
        </div>
      ))}
    </div>
  );
}

export function EditorPane() {
  const activeTab = useStore((s) => s.activeTab());
  const updateContent = useStore((s) => s.updateContent);
  const saveActive = useStore((s) => s.saveActive);
  const completionDisposable = useRef<IDisposable | null>(null);

  const handleMount: OnMount = useCallback(
    (editorInstance: editor.IStandaloneCodeEditor, monaco: Monaco) => {
      // Ctrl/Cmd+S saves the active file (and prevents the browser dialog).
      editorInstance.addCommand(
        monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS,
        () => {
          void saveActive();
        },
      );

      // Register an inline-completion provider that asks the backend for a
      // fill-in-the-middle continuation around the cursor. Registered once.
      if (!completionDisposable.current) {
        completionDisposable.current = monaco.languages.registerInlineCompletionsProvider(
          { pattern: "**" },
          {
            async provideInlineCompletions(model, position) {
              const offset = model.getOffsetAt(position);
              const full = model.getValue();
              const prefix = full.slice(0, offset);
              const suffix = full.slice(offset);
              // Skip trivial contexts to avoid noisy requests.
              if (prefix.trim().length === 0) {
                return { items: [] };
              }
              try {
                const text = await completeOnce({
                  prefix,
                  suffix,
                  language: model.getLanguageId(),
                  path: useStore.getState().activePath ?? "",
                  max_tokens: 128,
                });
                if (!text) return { items: [] };
                return {
                  items: [
                    {
                      insertText: text,
                      range: new monaco.Range(
                        position.lineNumber,
                        position.column,
                        position.lineNumber,
                        position.column,
                      ),
                    },
                  ],
                };
              } catch {
                return { items: [] };
              }
            },
            freeInlineCompletions() {
              /* nothing to dispose per-result */
            },
          },
        );
      }
    },
    [saveActive],
  );

  if (!activeTab) {
    return (
      <div className="editor-pane empty">
        <div className="editor-empty">
          <h2>aiforge</h2>
          <p>Open a file from the explorer to start editing.</p>
          <p className="hint">
            Press <kbd>Cmd/Ctrl</kbd>+<kbd>K</kbd> to issue an AI edit · type to
            get inline AI completions.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="editor-pane">
      <Tabs />
      <div className="editor-host">
        <Editor
          path={activeTab.path}
          language={languageForPath(activeTab.path)}
          value={activeTab.content}
          theme="vs-dark"
          onMount={handleMount}
          onChange={(value) => updateContent(activeTab.path, value ?? "")}
          options={{
            fontSize: 13,
            minimap: { enabled: true },
            inlineSuggest: { enabled: true },
            automaticLayout: true,
            scrollBeyondLastLine: false,
            tabSize: 2,
          }}
        />
      </div>
    </div>
  );
}
