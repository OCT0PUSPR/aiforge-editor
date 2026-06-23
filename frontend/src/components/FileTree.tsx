/**
 * Sidebar file tree. Loads `/api/tree` and lets the user open files. Renders
 * the recursive `TreeNode` structure with collapsible directories.
 */
import { useEffect, useState } from "react";
import { useStore } from "../store";
import type { TreeNode } from "../api/client";

function DirNode({ node, depth }: { node: TreeNode; depth: number }) {
  const [open, setOpen] = useState(depth < 1);
  const children = node.children ?? [];
  return (
    <div>
      <div
        className="tree-row tree-dir"
        style={{ paddingLeft: depth * 12 + 8 }}
        onClick={() => setOpen((o) => !o)}
      >
        <span className="tree-caret">{open ? "▾" : "▸"}</span>
        <span className="tree-icon">📁</span>
        <span className="tree-name">{node.name || "/"}</span>
      </div>
      {open &&
        children.map((child) =>
          child.type === "dir" ? (
            <DirNode key={child.path} node={child} depth={depth + 1} />
          ) : (
            <FileNode key={child.path} node={child} depth={depth + 1} />
          ),
        )}
    </div>
  );
}

function FileNode({ node, depth }: { node: TreeNode; depth: number }) {
  const openFile = useStore((s) => s.openFile);
  const activePath = useStore((s) => s.activePath);
  const isDirty = useStore((s) => s.isDirty);
  const active = activePath === node.path;
  return (
    <div
      className={`tree-row tree-file${active ? " active" : ""}`}
      style={{ paddingLeft: depth * 12 + 20 }}
      onClick={() => openFile(node.path)}
      title={node.path}
    >
      <span className="tree-icon">📄</span>
      <span className="tree-name">{node.name}</span>
      {isDirty(node.path) && <span className="tree-dirty">●</span>}
    </div>
  );
}

export function FileTree() {
  const tree = useStore((s) => s.tree);
  const loadTree = useStore((s) => s.loadTree);
  const buildIndex = useStore((s) => s.buildIndex);

  useEffect(() => {
    loadTree();
  }, [loadTree]);

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <span>EXPLORER</span>
        <div className="sidebar-actions">
          <button title="Refresh tree" onClick={() => loadTree()}>
            ⟳
          </button>
          <button title="Rebuild RAG index" onClick={() => buildIndex()}>
            ⚡
          </button>
        </div>
      </div>
      <div className="tree-scroll">
        {tree ? (
          tree.children?.length ? (
            tree.children.map((child) =>
              child.type === "dir" ? (
                <DirNode key={child.path} node={child} depth={0} />
              ) : (
                <FileNode key={child.path} node={child} depth={0} />
              ),
            )
          ) : (
            <div className="tree-empty">workspace is empty</div>
          )
        ) : (
          <div className="tree-empty">loading…</div>
        )}
      </div>
    </aside>
  );
}
