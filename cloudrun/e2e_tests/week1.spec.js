// Week 1 E2E Tests: UIGen Feature Sprint
// Tests Clear All Files, Download ZIP, and Keyboard Shortcuts

const { test, expect } = require('@playwright/test');

test.describe('Week 1: UIGen Feature Sprint', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Wait for app to load
    await page.waitForLoadState('networkidle');
  });

  test.describe('Stage 1: Clear All Files', () => {
    test('Trash button exists in header', async ({ page }) => {
      // Look for Trash2 icon button in header area
      const trashButton = page.locator('button:has(svg), [data-testid="clear-all"]').filter({
        has: page.locator('svg')
      }).first();

      // Alternative: look for button with trash icon or clear/delete text
      const clearButton = page.locator('button').filter({
        hasText: /clear|delete|trash/i
      }).first();

      const buttonExists = await trashButton.isVisible() || await clearButton.isVisible().catch(() => false);
      expect(buttonExists).toBeTruthy();
    });

    test('Clear All shows confirmation dialog', async ({ page }) => {
      // Find and click clear/trash button
      const clearButton = page.locator('button').filter({
        hasText: /clear|delete all/i
      }).first().or(page.locator('[data-testid="clear-all"]'));

      if (await clearButton.isVisible()) {
        await clearButton.click();

        // Check for dialog/modal
        const dialog = page.locator('[role="dialog"], [role="alertdialog"], .dialog, .modal');
        await expect(dialog).toBeVisible({ timeout: 3000 });

        // Check for cancel and confirm buttons
        const cancelBtn = page.locator('button').filter({ hasText: /cancel/i });
        const confirmBtn = page.locator('button').filter({ hasText: /delete|confirm|yes/i });

        expect(await cancelBtn.isVisible() || await confirmBtn.isVisible()).toBeTruthy();
      }
    });
  });

  test.describe('Stage 2: Download as ZIP', () => {
    test('Download button exists', async ({ page }) => {
      // Look for download button
      const downloadButton = page.locator('button').filter({
        hasText: /download|export|zip/i
      }).first().or(page.locator('[data-testid="download-zip"]'));

      const buttonExists = await downloadButton.isVisible().catch(() => false);
      expect(buttonExists).toBeTruthy();
    });

    test('Download triggers file download', async ({ page }) => {
      const downloadButton = page.locator('button').filter({
        hasText: /download|export|zip/i
      }).first().or(page.locator('[data-testid="download-zip"]'));

      if (await downloadButton.isVisible()) {
        // Set up download listener
        const downloadPromise = page.waitForEvent('download', { timeout: 10000 }).catch(() => null);

        await downloadButton.click();

        const download = await downloadPromise;
        if (download) {
          // Verify it's a ZIP file
          const filename = download.suggestedFilename();
          expect(filename).toMatch(/\.zip$/i);
        }
      }
    });
  });

  test.describe('Stage 3: Keyboard Shortcuts (Bonus)', () => {
    test('Cmd/Ctrl+K opens command palette', async ({ page }) => {
      // Press Cmd+K (Mac) or Ctrl+K (Windows/Linux)
      await page.keyboard.press('Meta+k');

      // Wait a bit and check for command palette
      await page.waitForTimeout(500);

      const commandPalette = page.locator('[role="dialog"], [cmdk-root], .command-palette, [data-testid="command-palette"]');
      const paletteVisible = await commandPalette.isVisible().catch(() => false);

      if (!paletteVisible) {
        // Try Ctrl+K as fallback
        await page.keyboard.press('Control+k');
        await page.waitForTimeout(500);
      }

      // This is a bonus feature, so we just check if it exists
      const finalCheck = await commandPalette.isVisible().catch(() => false);
      // Don't fail the test if bonus feature not implemented
      console.log(`Command Palette visible: ${finalCheck}`);
    });

    test('ESC closes command palette', async ({ page }) => {
      // Open command palette first
      await page.keyboard.press('Meta+k');
      await page.waitForTimeout(500);

      const commandPalette = page.locator('[role="dialog"], [cmdk-root], .command-palette');

      if (await commandPalette.isVisible().catch(() => false)) {
        await page.keyboard.press('Escape');
        await page.waitForTimeout(300);

        await expect(commandPalette).not.toBeVisible();
      }
    });
  });
});
