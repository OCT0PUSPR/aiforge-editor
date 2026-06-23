/**
 * Modal previewing a proposed agentic edit as a colourised unified diff, with
 * accept (apply to workspace) / reject controls.
 */
import { useStore } from "../store";

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

  const lines = preview.diff.split("\n").map(classify);
  const noChange = preview.original === preview.newContent;

  return (
    <div className="modal-overlay" onClick={() => setDiffPreview(null)}>
      <div className="diff-modal" onClick={(e) => e.stopPropagation()}>
        <div className="diff-head">
          <div>
            <strong>Proposed edit</strong> — {preview.path}
            <div className="diff-instruction">“{preview.instruction}”</div>
          </div>
          <button className="icon-btn" onClick={() => setDiffPreview(null)}>
            ×
          </button>
        </div>
        <div className="diff-body">
          {noChange ? (
            <div className="diff-empty">The model proposed no changes.</div>
          ) : (
            <pre className="diff-pre">
              {lines.map((l, i) => (
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
          <button
            className="btn-primary"
            disabled={noChange}
            onClick={() => void applyPreview()}
          >
            Accept &amp; Apply
          </button>
        </div>
      </div>
    </div>
  );
}
