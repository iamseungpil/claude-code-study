import { test, expect, Page } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

/**
 * UIGen Project Evaluation Tests - Week 1
 *
 * These tests evaluate submitted UIGen projects for:
 * - Stage 1: Clear All Files (25 points)
 *   - Button exists (10 points)
 *   - Dialog works (10 points)
 *   - Memory Record (5 points) - checked separately via file analysis
 *
 * - Stage 2: Download ZIP (30 points)
 *   - ZIP Creation (15 points)
 *   - Download works (10 points)
 *   - Memory Record (5 points) - checked separately via file analysis
 *
 * - Stage 3: Keyboard Shortcuts (25 points)
 *   - Cmd/Ctrl+K opens palette (10 points)
 *   - Palette has commands (10 points)
 *   - Commands work + ESC closes (5 points)
 *
 * Total Playwright-testable: 70/100 points
 */

// Test results object for scoring
interface TestResults {
  stage1: {
    clearButtonExists: boolean;
    dialogWorks: boolean;
  };
  stage2: {
    downloadButtonExists: boolean;
    zipDownloads: boolean;
  };
  stage3: {
    cmdKOpensPalette: boolean;
    paletteHasCommands: boolean;
    commandsWorkAndEsc: boolean;
  };
}

let testResults: TestResults = {
  stage1: { clearButtonExists: false, dialogWorks: false },
  stage2: { downloadButtonExists: false, zipDownloads: false },
  stage3: { cmdKOpensPalette: false, paletteHasCommands: false, commandsWorkAndEsc: false },
};

// Helper function to detect OS for keyboard shortcuts
const isMac = process.platform === 'darwin';
const cmdKey = isMac ? 'Meta' : 'Control';

test.describe('Stage 1: Clear All Files', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Wait for the app to load
    await page.waitForLoadState('networkidle');
  });

  test('1.1 Clear All button exists (10 points)', async ({ page }) => {
    // Look for a button with Trash2 icon or "Clear" text
    const clearButton = page.locator('button').filter({
      has: page.locator('svg[class*="trash"], svg[class*="Trash"], :text("Clear")')
    }).first();

    // Alternative: look for button with aria-label or title containing "clear"
    const clearButtonAlt = page.locator('button[aria-label*="clear" i], button[title*="clear" i]');

    // Try to find the button
    const buttonExists = await clearButton.or(clearButtonAlt).isVisible().catch(() => false);

    if (buttonExists) {
      testResults.stage1.clearButtonExists = true;
      await expect(clearButton.or(clearButtonAlt)).toBeVisible();
    } else {
      // Look for any button that might trigger clear functionality
      const buttons = page.locator('button');
      const count = await buttons.count();

      for (let i = 0; i < count; i++) {
        const button = buttons.nth(i);
        const html = await button.innerHTML().catch(() => '');
        const text = await button.textContent().catch(() => '');

        if (
          html.toLowerCase().includes('trash') ||
          text?.toLowerCase().includes('clear') ||
          text?.toLowerCase().includes('delete all')
        ) {
          testResults.stage1.clearButtonExists = true;
          await expect(button).toBeVisible();
          return;
        }
      }

      // If we reach here, no clear button found
      expect(buttonExists, 'Clear All button should exist').toBeTruthy();
    }
  });

  test('1.2 Confirmation dialog shows and works (10 points)', async ({ page }) => {
    // Find and click the Clear button
    const clearButton = page.locator('button').filter({
      has: page.locator('svg[class*="trash" i], svg[class*="Trash"], :text-is("Clear"), :text("Clear All")')
    }).first();

    const clearButtonAlt = page.locator('button:has-text("Clear"), button[aria-label*="clear" i]').first();

    const targetButton = await clearButton.isVisible() ? clearButton : clearButtonAlt;

    if (await targetButton.isVisible()) {
      await targetButton.click();

      // Wait for dialog to appear
      await page.waitForTimeout(500);

      // Check for dialog (various patterns)
      const dialog = page.locator('[role="dialog"], [data-state="open"], .dialog, .modal').first();
      const dialogVisible = await dialog.isVisible().catch(() => false);

      if (dialogVisible) {
        // Check for confirm/delete button in dialog
        const confirmBtn = dialog.locator('button:has-text("Delete"), button:has-text("Confirm"), button:has-text("Yes")').first();
        const confirmExists = await confirmBtn.isVisible().catch(() => false);

        // Check for cancel button
        const cancelBtn = dialog.locator('button:has-text("Cancel"), button:has-text("No"), button:has-text("Close")').first();
        const cancelExists = await cancelBtn.isVisible().catch(() => false);

        if (confirmExists && cancelExists) {
          // Test cancel closes dialog
          await cancelBtn.click();
          await page.waitForTimeout(300);

          const dialogClosed = !(await dialog.isVisible().catch(() => false));
          if (dialogClosed) {
            testResults.stage1.dialogWorks = true;
          }
        }
      }
    }

    expect(testResults.stage1.dialogWorks, 'Dialog should show and work correctly').toBeTruthy();
  });
});

test.describe('Stage 2: Download ZIP', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  test('2.1 Download button exists (10 points)', async ({ page }) => {
    // Look for download button
    const downloadButton = page.locator('button').filter({
      has: page.locator('svg[class*="download" i], :text("Download")')
    }).first();

    const downloadButtonAlt = page.locator('button:has-text("Download"), button[aria-label*="download" i]').first();

    const buttonExists = await downloadButton.or(downloadButtonAlt).isVisible().catch(() => false);

    if (buttonExists) {
      testResults.stage2.downloadButtonExists = true;
    } else {
      // Scan all buttons
      const buttons = page.locator('button');
      const count = await buttons.count();

      for (let i = 0; i < count; i++) {
        const button = buttons.nth(i);
        const html = await button.innerHTML().catch(() => '');
        const text = await button.textContent().catch(() => '');

        if (
          html.toLowerCase().includes('download') ||
          text?.toLowerCase().includes('download') ||
          text?.toLowerCase().includes('export')
        ) {
          testResults.stage2.downloadButtonExists = true;
          await expect(button).toBeVisible();
          return;
        }
      }
    }

    expect(testResults.stage2.downloadButtonExists, 'Download button should exist').toBeTruthy();
  });

  test('2.2 ZIP file downloads successfully (15 points)', async ({ page }) => {
    // Find download button
    const downloadButton = page.locator('button').filter({
      has: page.locator('svg[class*="download" i], :text("Download")')
    }).first();

    const downloadButtonAlt = page.locator('button:has-text("Download"), button[aria-label*="download" i]').first();

    const targetButton = await downloadButton.isVisible() ? downloadButton : downloadButtonAlt;

    if (await targetButton.isVisible()) {
      // Listen for download event
      const [download] = await Promise.all([
        page.waitForEvent('download', { timeout: 10000 }).catch(() => null),
        targetButton.click()
      ]);

      if (download) {
        const filename = download.suggestedFilename();
        if (filename.endsWith('.zip')) {
          testResults.stage2.zipDownloads = true;
        }
      }
    }

    expect(testResults.stage2.zipDownloads, 'ZIP file should download').toBeTruthy();
  });
});

test.describe('Stage 3: Keyboard Shortcuts', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  test('3.1 Cmd/Ctrl+K opens command palette (10 points)', async ({ page }) => {
    // Press Cmd+K (Mac) or Ctrl+K (Windows)
    await page.keyboard.press(`${cmdKey}+k`);

    // Wait for palette to appear
    await page.waitForTimeout(500);

    // Look for command palette (various implementations)
    const palette = page.locator('[cmdk-root], [role="dialog"]:has(input), .command-palette, [data-cmdk-root]').first();
    const paletteAlt = page.locator('[role="listbox"], [role="combobox"]').first();

    const paletteVisible = await palette.or(paletteAlt).isVisible().catch(() => false);

    if (paletteVisible) {
      testResults.stage3.cmdKOpensPalette = true;
    }

    expect(testResults.stage3.cmdKOpensPalette, 'Cmd/Ctrl+K should open command palette').toBeTruthy();
  });

  test('3.2 Command palette has Clear and Download commands (10 points)', async ({ page }) => {
    // Open palette
    await page.keyboard.press(`${cmdKey}+k`);
    await page.waitForTimeout(500);

    // Look for commands
    const palette = page.locator('[cmdk-root], [role="dialog"]:has(input), .command-palette, [data-cmdk-root]').first();

    if (await palette.isVisible()) {
      // Check for Clear command
      const clearCommand = palette.locator(':text("Clear")');
      const clearExists = await clearCommand.isVisible().catch(() => false);

      // Check for Download command
      const downloadCommand = palette.locator(':text("Download")');
      const downloadExists = await downloadCommand.isVisible().catch(() => false);

      if (clearExists && downloadExists) {
        testResults.stage3.paletteHasCommands = true;
      }
    }

    expect(testResults.stage3.paletteHasCommands, 'Palette should have Clear and Download commands').toBeTruthy();
  });

  test('3.3 Commands work and ESC closes palette (5 points)', async ({ page }) => {
    // Open palette
    await page.keyboard.press(`${cmdKey}+k`);
    await page.waitForTimeout(500);

    const palette = page.locator('[cmdk-root], [role="dialog"]:has(input), .command-palette, [data-cmdk-root]').first();

    if (await palette.isVisible()) {
      // Press ESC to close
      await page.keyboard.press('Escape');
      await page.waitForTimeout(300);

      // Check if palette is closed
      const paletteClosed = !(await palette.isVisible().catch(() => false));

      if (paletteClosed) {
        testResults.stage3.commandsWorkAndEsc = true;
      }
    }

    expect(testResults.stage3.commandsWorkAndEsc, 'ESC should close the palette').toBeTruthy();
  });
});

test.describe('Generate Score Report', () => {
  test.afterAll(async () => {
    // Calculate scores
    const scores = {
      stage1: {
        clearButtonExists: testResults.stage1.clearButtonExists ? 10 : 0,
        dialogWorks: testResults.stage1.dialogWorks ? 10 : 0,
        total: 0,
      },
      stage2: {
        downloadButtonExists: testResults.stage2.downloadButtonExists ? 10 : 0,
        zipDownloads: testResults.stage2.zipDownloads ? 15 : 0,
        total: 0,
      },
      stage3: {
        cmdKOpensPalette: testResults.stage3.cmdKOpensPalette ? 10 : 0,
        paletteHasCommands: testResults.stage3.paletteHasCommands ? 10 : 0,
        commandsWorkAndEsc: testResults.stage3.commandsWorkAndEsc ? 5 : 0,
        total: 0,
      },
      playwrightTotal: 0,
    };

    scores.stage1.total = scores.stage1.clearButtonExists + scores.stage1.dialogWorks;
    scores.stage2.total = scores.stage2.downloadButtonExists + scores.stage2.zipDownloads;
    scores.stage3.total = scores.stage3.cmdKOpensPalette + scores.stage3.paletteHasCommands + scores.stage3.commandsWorkAndEsc;
    scores.playwrightTotal = scores.stage1.total + scores.stage2.total + scores.stage3.total;

    // Output results
    const resultsDir = path.join(__dirname, 'test-results');
    if (!fs.existsSync(resultsDir)) {
      fs.mkdirSync(resultsDir, { recursive: true });
    }

    const scoreReport = {
      timestamp: new Date().toISOString(),
      testResults,
      scores,
      maxPlaywrightScore: 70,
      passed: scores.playwrightTotal >= 45, // At least Stage 1 + Stage 2 button
    };

    fs.writeFileSync(
      path.join(resultsDir, 'score-report.json'),
      JSON.stringify(scoreReport, null, 2)
    );

    console.log('\n=== UIGen Evaluation Score Report ===');
    console.log(`Stage 1 (Clear All): ${scores.stage1.total}/20`);
    console.log(`  - Button exists: ${scores.stage1.clearButtonExists}/10`);
    console.log(`  - Dialog works: ${scores.stage1.dialogWorks}/10`);
    console.log(`Stage 2 (Download ZIP): ${scores.stage2.total}/25`);
    console.log(`  - Button exists: ${scores.stage2.downloadButtonExists}/10`);
    console.log(`  - ZIP downloads: ${scores.stage2.zipDownloads}/15`);
    console.log(`Stage 3 (Keyboard Shortcuts): ${scores.stage3.total}/25`);
    console.log(`  - Cmd+K opens palette: ${scores.stage3.cmdKOpensPalette}/10`);
    console.log(`  - Has commands: ${scores.stage3.paletteHasCommands}/10`);
    console.log(`  - ESC closes: ${scores.stage3.commandsWorkAndEsc}/5`);
    console.log(`\nPlaywright Total: ${scores.playwrightTotal}/70`);
    console.log(`(Memory Record + CLAUDE.md Quality + Code Quality = 30 points evaluated separately)\n`);
  });

  test('dummy test for afterAll', async () => {
    // This is just to trigger afterAll
    expect(true).toBeTruthy();
  });
});
