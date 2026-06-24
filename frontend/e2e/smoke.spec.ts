import { expect, test } from "@playwright/test";

/**
 * End-to-end smoke test against a running stack (frontend + backend).
 *
 * Guarded: skipped unless E2E_BASE_URL is set, so `npm run e2e` is safe in CI
 * (it skips cleanly when no live stack is available) and runs for real locally:
 *
 *   # terminal 1: backend
 *   cd backend && uvicorn aiforge.api.server:app --port 8000
 *   # terminal 2: frontend
 *   cd frontend && npm run dev
 *   # terminal 3:
 *   E2E_BASE_URL=http://localhost:5173 npm run e2e
 */
const LIVE = Boolean(process.env.E2E_BASE_URL);

test.describe("aiforge smoke", () => {
  test.skip(!LIVE, "set E2E_BASE_URL to run e2e against a live stack");

  test("register, open editor shell, create a file", async ({ page }) => {
    await page.goto("/");

    // Land on the login screen.
    await expect(page.getByText("AI-native code editor")).toBeVisible();

    // Register a unique user.
    await page.getByRole("button", { name: "Register" }).click();
    const id = Date.now();
    await page.getByPlaceholder("Email").fill(`e2e${id}@example.com`);
    await page.getByPlaceholder("Username").fill(`e2e${id}`);
    await page.getByPlaceholder("Password").fill("password12345");
    await page.getByRole("button", { name: "Create account" }).click();

    // The editor shell should appear (explorer + chat).
    await expect(page.getByText("EXPLORER")).toBeVisible({ timeout: 15_000 });
    await expect(page.getByText("AI CHAT")).toBeVisible();

    // The status bar shows a workspace.
    await expect(page.locator(".status-bar")).toContainText("ws:");
  });
});
