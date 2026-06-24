# Security

This document describes aiforge-editor's security model, with emphasis on the
**workspace sandbox** (the core safety guarantee) and how to report issues.

## Reporting a vulnerability

Please open a private security advisory or email the maintainers rather than
filing a public issue. Include reproduction steps and affected versions.

## Workspace sandbox guarantees

Every file operation goes through `aiforge.workspace.files.Workspace`, which
confines all reads and writes to a single root directory. The resolver
(`Workspace.resolve`) enforces, in order:

1. **Reject NUL bytes and control characters** in the path.
2. **Reject absolute paths.** A leading `/` is treated as *workspace-relative*,
   so `/etc/passwd` maps to `<root>/etc/passwd` (inside the sandbox), never the
   filesystem root.
3. **Reject any `..` component** before resolution.
4. **Canonicalize and re-check.** The resolved real path must still be inside
   the canonical root — this catches **symlink escapes** (a symlink whose target
   leaves the root) and any residual traversal.

These properties are covered by `tests/test_workspace.py` (parametrized
traversal payloads, a symlink-escape test, and leading-slash handling).

### Tenant isolation

Each workspace is provisioned with a distinct, server-generated directory
(`<data_dir>/workspaces/<root_dir>`). Because every tenant's sandbox has a
**separate root**, one user's path resolution can never reach another's files.
Workspace ownership is checked on every request (`get_workspace` returns 404 for
a workspace the caller doesn't own). See `tests/test_isolation.py`.

### Quotas

Writes enforce per-workspace quotas (max file size, max total bytes, max file
count) and refuse binary file types, so a workspace cannot exhaust disk or be
used to stage arbitrary binaries.

## Authentication & authorization

- **Passwords** are hashed with bcrypt (SHA-256 pre-hash to lift the 72-byte
  limit; the Django `bcrypt_sha256` pattern). Plaintext is never stored.
- **JWT** access (short-lived) and refresh (longer-lived) tokens are HS256-signed
  with `AIFORGE_JWT_SECRET`. Set a stable 32+ byte secret in production
  (`openssl rand -hex 32`); otherwise a random per-process secret is used and
  tokens don't survive a restart.
- **API keys** are shown once at creation and stored only as a SHA-256 hash.
  They authenticate the same endpoints as a JWT and can be revoked.
- Every workspace/file/AI endpoint requires authentication; ownership is
  enforced per request.

## Transport & headers

The middleware sets `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`,
a restrictive `Content-Security-Policy`, and `Referrer-Policy`. CORS is an
explicit allowlist (`AIFORGE_CORS_ORIGINS`), not a wildcard with credentials.

## Rate limiting

General and AI endpoints are rate-limited per identity (user id or client IP)
with a token-bucket limiter, optionally Redis-backed so limits hold across
replicas. Limits are configurable (`AIFORGE_RATE_LIMIT_RPM`,
`AIFORGE_RATE_LIMIT_AI_RPM`).

## Input validation & limits

Request bodies are size-capped (`AIFORGE_MAX_REQUEST_BYTES`); Pydantic schemas
bound the length of every string field (paths, instructions, chat content);
file paths are validated by the sandbox before any filesystem call.

## Secrets

No secrets are ever hardcoded. Vendor keys (`ANTHROPIC_API_KEY`, `HF_TOKEN`) and
`AIFORGE_JWT_SECRET` come from the environment only; `.env.example` ships
placeholders. `.gitignore` excludes `.env`, the SQLite DB, workspace data, and
model weights. `detect-private-key` runs in pre-commit.

## Static analysis

`bandit` runs in CI and pre-commit over `aiforge/` (the MD5 in the RAG hashing
embedder is explicitly `usedforsecurity=False` — it's feature hashing, not
security). `pip-audit` runs in CI (informational) to flag vulnerable
dependencies.

## Threat-model notes

- The editor executes **no** user code server-side; it only reads/writes text
  files in a sandbox. There is no code-execution surface in the backend.
- LLM output that is "applied to a file" is always shown as a reviewable diff
  first (single-file via Monaco's diff editor, multi-file via a unified diff)
  and only written on explicit accept.
- The mock backend makes the entire product runnable with **no** external
  network calls and **no** keys, which is also the default CI path.
