/**
 * Sidebar file tree with create / rename / delete. Loads the workspace tree and
 * lets the user open files; right-click (or the row actions) for CRUD.
 */
import { useState } from "react";
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
  const renameFile = useStore((s) => s.renameFile);
  const deleteFile = useStore((s) => s.deleteFile);
  const activePath = useStore((s) => s.activePath);
  const isDirty = useStore((s) => s.isDirty);
  const active = activePath === node.path;

  const onRename = (e: React.MouseEvent) => {
    e.stopPropagation();
    const dst = window.prompt("Rename to:", node.path);
    if (dst && dst !== node.path) void renameFile(node.path, dst);
  };
  const onDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (window.confirm(`Delete ${node.path}?`)) void deleteFile(node.path);
  };

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
      <span className="tree-actions">
        <button title="Rename" onClick={onRename}>
          ✎
        </button>
        <button title="Delete" onClick={onDelete}>
          🗑
        </button>
      </span>
    </div>
  );
}

export function FileTree() {
  const tree = useStore((s) => s.tree);
  const loadTree = useStore((s) => s.loadTree);
  const buildIndex = useStore((s) => s.buildIndex);
  const createFile = useStore((s) => s.createFile);

  const onNew = () => {
    const path = window.prompt("New file path (e.g. src/util.py):");
    if (path) void createFile(path);
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <span>EXPLORER</span>
        <div className="sidebar-actions">
          <button title="New file" onClick={onNew}>
            ✚
          </button>
          <button title="Refresh tree" onClick={() => void loadTree()}>
            ⟳
          </button>
          <button title="Rebuild RAG index" onClick={() => void buildIndex()}>
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
            <div className="tree-empty">workspace is empty — create a file</div>
          )
        ) : (
          <div className="tree-empty">loading…</div>
        )}
      </div>
    </aside>
  );
}
