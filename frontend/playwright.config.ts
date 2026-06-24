import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright e2e config. Guarded: e2e tests are skipped unless E2E_BASE_URL is
 * set (so CI can run them headless against a live stack, or skip cleanly).
 */
export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  retries: 0,
  use: {
    baseURL: process.env.E2E_BASE_URL || "http://localhost:5173",
    headless: true,
    trace: "on-first-retry",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
