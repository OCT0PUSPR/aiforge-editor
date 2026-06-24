/**
 * Diff preview modal. Single-file edits render in Monaco's side-by-side
 * DiffEditor; multi-file edits render a colourised unified diff. Accept applies
 * the change to the workspace; reject discards it.
 */
import { DiffEditor } from "@monaco-editor/react";
import { useStore } from "../store";
import { languageForPath } from "../language";

interface DiffLine {
  text: string;
  kind: "add" | "del" | "ctx" | "hunk" | "meta";
}
function classify(line: string): DiffLine {
  if (line.startsWith("+++") || line.startsWith("---")) return { text: line, kind: "meta" };
  if (line.startsWith("@@")) return { text: line, kind: "hunk" };
  if (line.startsWith("+")) return { text: line, kind: "add" };
  if (line.startsWith("-")) return { text: line, kind: "del" };
  return { text: line, kind: "ctx" };
}

export function DiffModal() {
  const preview = useStore((s) => s.diffPreview);
  const setDiffPreview = useStore((s) => s.setDiffPreview);
  const applyPreview = useStore((s) => s.applyPreview);
  if (!preview) return null;

  const single = !preview.multifile;
  const noChange = single
    ? preview.original === preview.newContent
    : (preview.files ?? []).length === 0;

  return (
    <div className="modal-overlay" onClick={() => setDiffPreview(null)}>
      <div className="diff-modal" onClick={(e) => e.stopPropagation()}>
        <div className="diff-head">
          <div>
            <strong>Proposed edit</strong> —{" "}
            {single ? preview.paths[0] : `${preview.files?.length ?? 0} file(s)`}
            <div className="diff-instruction">“{preview.instruction}”</div>
          </div>
          <button className="icon-btn" onClick={() => setDiffPreview(null)}>
            ×
          </button>
        </div>
        <div className="diff-body">
          {noChange ? (
            <div className="diff-empty">The model proposed no changes.</div>
          ) : single ? (
            <DiffEditor
              original={preview.original}
              modified={preview.newContent ?? ""}
              language={languageForPath(preview.paths[0])}
              theme="vs-dark"
              options={{
                readOnly: true,
                renderSideBySide: true,
                automaticLayout: true,
                minimap: { enabled: false },
                fontSize: 12,
              }}
              height="100%"
            />
          ) : (
            <pre className="diff-pre">
              {preview.diff.split("\n").map(classify).map((l, i) => (
                <div key={i} className={`diff-line diff-${l.kind}`}>
                  {l.text || " "}
                </div>
              ))}
            </pre>
          )}
        </div>
        <div className="diff-actions">
          <button className="btn-secondary" onClick={() => setDiffPreview(null)}>
            Reject
          </button>
          <button className="btn-primary" disabled={noChange} onClick={() => void applyPreview()}>
            Accept &amp; Apply
          </button>
        </div>
      </div>
    </div>
  );
}
