/**
 * Global application state (Zustand).
 *
 * Holds the file tree, open tabs, the active tab, and the dirty flag per tab.
 * Components subscribe to slices of this store; the editor and file tree stay
 * in sync through it.
 */
import { create } from "zustand";
import * as api from "./api/client";
import type { TreeNode } from "./api/client";

export interface OpenTab {
  path: string;
  /** Last-saved content (what's on disk). */
  saved: string;
  /** Current editor content (may differ from `saved`). */
  content: string;
}

export interface DiffPreview {
  path: string;
  diff: string;
  newContent: string;
  original: string;
  instruction: string;
}

interface AppState {
  tree: TreeNode | null;
  tabs: OpenTab[];
  activePath: string | null;
  status: string;
  diffPreview: DiffPreview | null;
  indexStats: api.IndexStats | null;

  // derived
  activeTab: () => OpenTab | null;
  isDirty: (path: string) => boolean;

  // actions
  loadTree: () => Promise<void>;
  openFile: (path: string) => Promise<void>;
  closeTab: (path: string) => void;
  setActive: (path: string) => void;
  updateContent: (path: string, content: string) => void;
  saveActive: () => Promise<void>;
  saveTab: (path: string) => Promise<void>;
  setStatus: (status: string) => void;
  setDiffPreview: (preview: DiffPreview | null) => void;
  buildIndex: () => Promise<void>;
  applyPreview: () => Promise<void>;
}

export const useStore = create<AppState>((set, get) => ({
  tree: null,
  tabs: [],
  activePath: null,
  status: "ready",
  diffPreview: null,
  indexStats: null,

  activeTab: () => {
    const { tabs, activePath } = get();
    return tabs.find((t) => t.path === activePath) ?? null;
  },

  isDirty: (path) => {
    const tab = get().tabs.find((t) => t.path === path);
    return tab ? tab.content !== tab.saved : false;
  },

  loadTree: async () => {
    set({ status: "loading tree…" });
    try {
      const tree = await api.getTree();
      set({ tree, status: "ready" });
    } catch (e) {
      set({ status: `error: ${(e as Error).message}` });
    }
  },

  openFile: async (path) => {
    const existing = get().tabs.find((t) => t.path === path);
    if (existing) {
      set({ activePath: path });
      return;
    }
    set({ status: `opening ${path}…` });
    try {
      const { content } = await api.getFile(path);
      set((s) => ({
        tabs: [...s.tabs, { path, saved: content, content }],
        activePath: path,
        status: "ready",
      }));
    } catch (e) {
      set({ status: `error: ${(e as Error).message}` });
    }
  },

  closeTab: (path) => {
    set((s) => {
      const tabs = s.tabs.filter((t) => t.path !== path);
      const activePath =
        s.activePath === path ? (tabs.length ? tabs[tabs.length - 1].path : null) : s.activePath;
      return { tabs, activePath };
    });
  },

  setActive: (path) => set({ activePath: path }),

  updateContent: (path, content) =>
    set((s) => ({
      tabs: s.tabs.map((t) => (t.path === path ? { ...t, content } : t)),
    })),

  saveTab: async (path) => {
    const tab = get().tabs.find((t) => t.path === path);
    if (!tab) return;
    set({ status: `saving ${path}…` });
    try {
      await api.saveFile(path, tab.content);
      set((s) => ({
        tabs: s.tabs.map((t) => (t.path === path ? { ...t, saved: t.content } : t)),
        status: `saved ${path}`,
      }));
    } catch (e) {
      set({ status: `error: ${(e as Error).message}` });
    }
  },

  saveActive: async () => {
    const path = get().activePath;
    if (path) await get().saveTab(path);
  },

  setStatus: (status) => set({ status }),

  setDiffPreview: (diffPreview) => set({ diffPreview }),

  buildIndex: async () => {
    set({ status: "indexing codebase…" });
    try {
      const stats = await api.buildIndex();
      set({ indexStats: stats, status: `indexed ${stats.files} files / ${stats.chunks} chunks` });
    } catch (e) {
      set({ status: `error: ${(e as Error).message}` });
    }
  },

  applyPreview: async () => {
    const preview = get().diffPreview;
    if (!preview) return;
    set({ status: `applying edit to ${preview.path}…` });
    try {
      const result = await api.applyEdit({
        path: preview.path,
        new_content: preview.newContent,
      });
      // Reflect the change in any open tab and refresh the tree.
      set((s) => ({
        tabs: s.tabs.map((t) =>
          t.path === preview.path
            ? { ...t, saved: result.new_content, content: result.new_content }
            : t,
        ),
        diffPreview: null,
        status: `applied edit to ${preview.path}`,
      }));
      // Make sure the file is open and the tree reflects new files.
      if (!get().tabs.find((t) => t.path === preview.path)) {
        await get().openFile(preview.path);
      }
      await get().loadTree();
    } catch (e) {
      set({ status: `error: ${(e as Error).message}` });
    }
  },
}));
