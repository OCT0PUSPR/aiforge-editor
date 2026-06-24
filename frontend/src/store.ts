/**
 * Global application state (Zustand).
 *
 * Covers auth/session, the active workspace, open editor tabs (with dirty
 * tracking), the file tree, RAG index stats, the agentic-edit diff preview,
 * toasts, and UI flags (command palette, settings). All file/AI calls are
 * scoped to the active workspace.
 */
import { create } from "zustand";
import * as api from "./api/client";
import type {
  EditHistoryItem,
  IndexStats,
  TreeNode,
  User,
  Workspace,
} from "./api/client";

export interface OpenTab {
  path: string;
  saved: string;
  content: string;
}

export interface Toast {
  id: number;
  kind: "info" | "success" | "error";
  message: string;
}

export interface DiffPreview {
  paths: string[];
  diff: string;
  // single-file
  newContent?: string;
  // multi-file
  files?: { path: string; new_content: string }[];
  original: string;
  instruction: string;
  multifile: boolean;
}

export interface Settings {
  topK: number;
  completionEnabled: boolean;
}

type Status = "init" | "unauthenticated" | "ready";

interface AppState {
  // session
  status: Status;
  user: User | null;
  workspaces: Workspace[];
  activeWorkspace: Workspace | null;

  // editor
  tree: TreeNode | null;
  tabs: OpenTab[];
  activePath: string | null;
  indexStats: IndexStats | null;
  history: EditHistoryItem[];

  // ui
  statusText: string;
  toasts: Toast[];
  diffPreview: DiffPreview | null;
  paletteOpen: boolean;
  settingsOpen: boolean;
  settings: Settings;

  // derived
  activeTab: () => OpenTab | null;
  isDirty: (path: string) => boolean;

  // toasts
  toast: (kind: Toast["kind"], message: string) => void;
  dismissToast: (id: number) => void;

  // session actions
  bootstrap: () => Promise<void>;
  login: (username: string, password: string) => Promise<void>;
  register: (email: string, username: string, password: string) => Promise<void>;
  logout: () => void;

  // workspace actions
  loadWorkspaces: () => Promise<void>;
  switchWorkspace: (id: string) => Promise<void>;
  createWorkspace: (name: string) => Promise<void>;

  // file actions
  loadTree: () => Promise<void>;
  openFile: (path: string) => Promise<void>;
  closeTab: (path: string) => void;
  setActive: (path: string) => void;
  updateContent: (path: string, content: string) => void;
  saveActive: () => Promise<void>;
  saveTab: (path: string) => Promise<void>;
  createFile: (path: string) => Promise<void>;
  renameFile: (src: string, dst: string) => Promise<void>;
  deleteFile: (path: string) => Promise<void>;

  // rag
  buildIndex: () => Promise<void>;

  // edits
  setDiffPreview: (p: DiffPreview | null) => void;
  applyPreview: () => Promise<void>;
  loadHistory: () => Promise<void>;
  undoEdit: (id: string) => Promise<void>;

  // ui
  setPaletteOpen: (open: boolean) => void;
  setSettingsOpen: (open: boolean) => void;
  setSettings: (s: Partial<Settings>) => void;
  setStatus: (text: string) => void;
}

let toastSeq = 1;

export const useStore = create<AppState>((set, get) => ({
  status: "init",
  user: null,
  workspaces: [],
  activeWorkspace: null,
  tree: null,
  tabs: [],
  activePath: null,
  indexStats: null,
  history: [],
  statusText: "ready",
  toasts: [],
  diffPreview: null,
  paletteOpen: false,
  settingsOpen: false,
  settings: { topK: 6, completionEnabled: true },

  activeTab: () => {
    const { tabs, activePath } = get();
    return tabs.find((t) => t.path === activePath) ?? null;
  },
  isDirty: (path) => {
    const tab = get().tabs.find((t) => t.path === path);
    return tab ? tab.content !== tab.saved : false;
  },

  toast: (kind, message) => {
    const id = toastSeq++;
    set((s) => ({ toasts: [...s.toasts, { id, kind, message }] }));
    setTimeout(() => get().dismissToast(id), kind === "error" ? 6000 : 3500);
  },
  dismissToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),

  // -- session --
  bootstrap: async () => {
    if (!api.tokens.access) {
      set({ status: "unauthenticated" });
      return;
    }
    try {
      const user = await api.auth.me();
      set({ user });
      await get().loadWorkspaces();
      set({ status: "ready" });
    } catch {
      api.tokens.clear();
      set({ status: "unauthenticated" });
    }
  },
  login: async (username, password) => {
    await api.auth.login(username, password);
    await get().bootstrap();
    get().toast("success", "Signed in");
  },
  register: async (email, username, password) => {
    await api.auth.register(email, username, password);
    await get().bootstrap();
    get().toast("success", "Account created");
  },
  logout: () => {
    api.auth.logout();
    set({
      status: "unauthenticated",
      user: null,
      workspaces: [],
      activeWorkspace: null,
      tabs: [],
      tree: null,
      activePath: null,
    });
  },

  // -- workspaces --
  loadWorkspaces: async () => {
    const ws = await api.workspaces.list();
    set({ workspaces: ws });
    const active = get().activeWorkspace;
    if (!active && ws.length) {
      await get().switchWorkspace(ws[0].id);
    }
  },
  switchWorkspace: async (id) => {
    const ws = get().workspaces.find((w) => w.id === id) ?? null;
    set({ activeWorkspace: ws, tabs: [], activePath: null, tree: null, indexStats: null });
    if (ws) {
      await get().loadTree();
      await get().buildIndex();
    }
  },
  createWorkspace: async (name) => {
    const ws = await api.workspaces.create(name);
    set((s) => ({ workspaces: [...s.workspaces, ws] }));
    await get().switchWorkspace(ws.id);
    get().toast("success", `Workspace "${name}" created`);
  },

  // -- files --
  loadTree: async () => {
    const ws = get().activeWorkspace;
    if (!ws) return;
    try {
      const tree = await api.files(ws.id).tree();
      set({ tree });
    } catch (e) {
      get().toast("error", `Failed to load files: ${(e as Error).message}`);
    }
  },
  openFile: async (path) => {
    const ws = get().activeWorkspace;
    if (!ws) return;
    if (get().tabs.find((t) => t.path === path)) {
      set({ activePath: path });
      return;
    }
    try {
      const { content } = await api.files(ws.id).read(path);
      set((s) => ({ tabs: [...s.tabs, { path, saved: content, content }], activePath: path }));
    } catch (e) {
      get().toast("error", `Cannot open ${path}: ${(e as Error).message}`);
    }
  },
  closeTab: (path) =>
    set((s) => {
      const tabs = s.tabs.filter((t) => t.path !== path);
      const activePath =
        s.activePath === path ? (tabs.length ? tabs[tabs.length - 1].path : null) : s.activePath;
      return { tabs, activePath };
    }),
  setActive: (path) => set({ activePath: path }),
  updateContent: (path, content) =>
    set((s) => ({ tabs: s.tabs.map((t) => (t.path === path ? { ...t, content } : t)) })),
  saveTab: async (path) => {
    const ws = get().activeWorkspace;
    const tab = get().tabs.find((t) => t.path === path);
    if (!ws || !tab) return;
    try {
      await api.files(ws.id).save(path, tab.content);
      set((s) => ({ tabs: s.tabs.map((t) => (t.path === path ? { ...t, saved: t.content } : t)) }));
      set({ statusText: `saved ${path}` });
    } catch (e) {
      get().toast("error", `Save failed: ${(e as Error).message}`);
    }
  },
  saveActive: async () => {
    const path = get().activePath;
    if (path) await get().saveTab(path);
  },
  createFile: async (path) => {
    const ws = get().activeWorkspace;
    if (!ws) return;
    try {
      await api.files(ws.id).create(path);
      await get().loadTree();
      await get().openFile(path);
      get().toast("success", `Created ${path}`);
    } catch (e) {
      get().toast("error", `Create failed: ${(e as Error).message}`);
    }
  },
  renameFile: async (src, dst) => {
    const ws = get().activeWorkspace;
    if (!ws) return;
    try {
      await api.files(ws.id).rename(src, dst);
      set((s) => ({
        tabs: s.tabs.map((t) => (t.path === src ? { ...t, path: dst } : t)),
        activePath: s.activePath === src ? dst : s.activePath,
      }));
      await get().loadTree();
      get().toast("success", `Renamed to ${dst}`);
    } catch (e) {
      get().toast("error", `Rename failed: ${(e as Error).message}`);
    }
  },
  deleteFile: async (path) => {
    const ws = get().activeWorkspace;
    if (!ws) return;
    try {
      await api.files(ws.id).remove(path);
      get().closeTab(path);
      await get().loadTree();
      get().toast("info", `Deleted ${path}`);
    } catch (e) {
      get().toast("error", `Delete failed: ${(e as Error).message}`);
    }
  },

  // -- rag --
  buildIndex: async () => {
    const ws = get().activeWorkspace;
    if (!ws) return;
    set({ statusText: "indexing codebase…" });
    try {
      const stats = await api.rag(ws.id).index();
      set({ indexStats: stats, statusText: `indexed ${stats.files} files / ${stats.chunks} chunks` });
    } catch (e) {
      get().toast("error", `Index failed: ${(e as Error).message}`);
    }
  },

  // -- edits --
  setDiffPreview: (diffPreview) => set({ diffPreview }),
  applyPreview: async () => {
    const ws = get().activeWorkspace;
    const preview = get().diffPreview;
    if (!ws || !preview) return;
    set({ statusText: "applying edit…" });
    try {
      if (preview.multifile) {
        await api.ai(ws.id).apply({
          diff: preview.diff,
          multifile: true,
          instruction: preview.instruction,
        });
        // Refresh any open tabs that were edited.
        for (const f of preview.files ?? []) {
          if (get().tabs.find((t) => t.path === f.path)) {
            set((s) => ({
              tabs: s.tabs.map((t) =>
                t.path === f.path ? { ...t, saved: f.new_content, content: f.new_content } : t,
              ),
            }));
          }
        }
      } else {
        await api.ai(ws.id).apply({
          path: preview.paths[0],
          new_content: preview.newContent,
          instruction: preview.instruction,
        });
        const path = preview.paths[0];
        const nc = preview.newContent ?? "";
        set((s) => ({
          tabs: s.tabs.map((t) => (t.path === path ? { ...t, saved: nc, content: nc } : t)),
        }));
        if (!get().tabs.find((t) => t.path === path)) await get().openFile(path);
      }
      set({ diffPreview: null, statusText: "edit applied" });
      get().toast("success", "Edit applied");
      await get().loadTree();
      await get().loadHistory();
    } catch (e) {
      get().toast("error", `Apply failed: ${(e as Error).message}`);
    }
  },
  loadHistory: async () => {
    const ws = get().activeWorkspace;
    if (!ws) return;
    try {
      const { history } = await api.ai(ws.id).history();
      set({ history });
    } catch {
      /* non-fatal */
    }
  },
  undoEdit: async (id) => {
    const ws = get().activeWorkspace;
    if (!ws) return;
    try {
      await api.ai(ws.id).undo(id);
      get().toast("info", "Edit undone");
      // Reload any open tabs (content may have changed on disk).
      const open = get().tabs.map((t) => t.path);
      set({ tabs: get().tabs.filter(() => false), activePath: null });
      for (const p of open) await get().openFile(p);
      await get().loadHistory();
    } catch (e) {
      get().toast("error", `Undo failed: ${(e as Error).message}`);
    }
  },

  // -- ui --
  setPaletteOpen: (paletteOpen) => set({ paletteOpen }),
  setSettingsOpen: (settingsOpen) => set({ settingsOpen }),
  setSettings: (s) => set((st) => ({ settings: { ...st.settings, ...s } })),
  setStatus: (statusText) => set({ statusText }),
}));
