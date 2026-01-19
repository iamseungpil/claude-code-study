import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright configuration for Week 1 UIGen Challenge tests
 *
 * Tests are organized into two categories:
 * 1. Site tests - Tests for the deployed study site (login, challenge participation)
 * 2. UIGen tests - Tests for evaluating submitted UIGen projects
 */
export default defineConfig({
  testDir: '.',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [
    ['html'],
    ['json', { outputFile: 'test-results/results.json' }]
  ],

  use: {
    // Base URL for the deployed site
    baseURL: process.env.SITE_URL || 'https://claude-code-study.pages.dev/frontend/',

    // Collect trace on failure
    trace: 'on-first-retry',

    // Screenshot on failure
    screenshot: 'only-on-failure',

    // Video recording
    video: 'retain-on-failure',
  },

  projects: [
    // Site tests - test login, challenge flow on deployed site
    {
      name: 'site-tests',
      testMatch: /site\.spec\.ts/,
      use: {
        ...devices['Desktop Chrome'],
        baseURL: process.env.SITE_URL || 'https://claude-code-study.pages.dev/frontend/',
      },
    },

    // UIGen evaluation tests - test submitted projects locally
    {
      name: 'uigen-evaluation',
      testMatch: /uigen\.spec\.ts/,
      use: {
        ...devices['Desktop Chrome'],
        // UIGen projects run on localhost
        baseURL: process.env.UIGEN_URL || 'http://localhost:3000',
      },
    },
  ],

  // Output directories
  outputDir: 'test-results/',
});
