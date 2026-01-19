import { test, expect } from '@playwright/test';

/**
 * Site Tests for Claude Code Study - Week 1
 *
 * These tests verify the deployed site functionality:
 * - Login/Signup flow
 * - Challenge participation flow
 * - Submission flow
 */

test.describe('Authentication Flow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should display login and signup buttons when not authenticated', async ({ page }) => {
    // Check login button exists
    const loginButton = page.locator('button').filter({ hasText: 'Login' }).first();
    await expect(loginButton).toBeVisible();

    // Check signup button exists
    const signupButton = page.locator('button').filter({ hasText: 'Sign Up' }).first();
    await expect(signupButton).toBeVisible();
  });

  test('should open login modal when clicking login button', async ({ page }) => {
    // Click login button
    await page.locator('#auth-buttons button').filter({ hasText: 'Login' }).click();

    // Check modal is visible
    const modal = page.locator('#login-modal');
    await expect(modal).not.toHaveClass(/hidden/);

    // Check form elements exist
    await expect(page.locator('#login-userid')).toBeVisible();
    await expect(page.locator('#login-password')).toBeVisible();
    await expect(page.locator('#login-form button[type="submit"]')).toBeVisible();
  });

  test('should show error message with invalid credentials', async ({ page }) => {
    // Open login modal
    await page.locator('#auth-buttons button').filter({ hasText: 'Login' }).click();

    // Fill in invalid credentials
    await page.locator('#login-userid').fill('invalid_user');
    await page.locator('#login-password').fill('wrong_password');

    // Submit form
    await page.locator('#login-form button[type="submit"]').click();

    // Wait for error message
    await expect(page.locator('#login-error')).toBeVisible();
    await expect(page.locator('#login-error')).toContainText(/Invalid|error/i);
  });

  test('should open signup modal when clicking signup button', async ({ page }) => {
    // Click signup button
    await page.locator('#auth-buttons button').filter({ hasText: 'Sign Up' }).click();

    // Check modal is visible
    const modal = page.locator('#signup-modal');
    await expect(modal).not.toHaveClass(/hidden/);

    // Check form elements exist
    await expect(page.locator('#signup-userid')).toBeVisible();
    await expect(page.locator('#signup-firstname')).toBeVisible();
    await expect(page.locator('#signup-lastname')).toBeVisible();
    await expect(page.locator('#signup-password')).toBeVisible();
    await expect(page.locator('#signup-confirm')).toBeVisible();
  });

  test('should show password mismatch error in signup', async ({ page }) => {
    // Open signup modal
    await page.locator('#auth-buttons button').filter({ hasText: 'Sign Up' }).click();

    // Fill in form with mismatched passwords
    await page.locator('#signup-userid').fill('test_user');
    await page.locator('#signup-firstname').fill('Test');
    await page.locator('#signup-lastname').fill('User');
    await page.locator('#signup-password').fill('password123');
    await page.locator('#signup-confirm').fill('different_password');

    // Submit form
    await page.locator('#signup-form button[type="submit"]').click();

    // Check for error message
    await expect(page.locator('#signup-error')).toBeVisible();
    await expect(page.locator('#signup-error')).toContainText(/match/i);
  });

  test('should close modal when clicking outside', async ({ page }) => {
    // Open login modal
    await page.locator('#auth-buttons button').filter({ hasText: 'Login' }).click();
    await expect(page.locator('#login-modal')).not.toHaveClass(/hidden/);

    // Click outside the modal (on the overlay)
    await page.locator('#login-modal').click({ position: { x: 10, y: 10 } });

    // Modal should be hidden
    await expect(page.locator('#login-modal')).toHaveClass(/hidden/);
  });
});

test.describe('Main Page Elements', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should display main page with all sections', async ({ page }) => {
    // Hero section
    await expect(page.locator('h1')).toContainText('Claude Code');
    await expect(page.locator('h1')).toContainText('Study Group');

    // Quick Submit section
    await expect(page.locator('h2').filter({ hasText: 'Quick Submit' })).toBeVisible();

    // Leaderboard section
    await expect(page.locator('h2').filter({ hasText: 'Live Leaderboard' })).toBeVisible();

    // Weekly Challenges section
    await expect(page.locator('h2').filter({ hasText: 'Weekly Challenges' })).toBeVisible();
  });

  test('should have week selector in Quick Submit', async ({ page }) => {
    const weekSelector = page.locator('#submit-week');
    await expect(weekSelector).toBeVisible();

    // Check all week options exist
    await expect(weekSelector.locator('option')).toHaveCount(5);
    await expect(weekSelector.locator('option').first()).toContainText('Week 1');
  });

  test('should show login warning when not authenticated', async ({ page }) => {
    const loginWarning = page.locator('#login-required-warning');
    await expect(loginWarning).toBeVisible();
    await expect(loginWarning).toContainText('login');
  });

  test('should have disabled submit button when not authenticated', async ({ page }) => {
    const submitButton = page.locator('#submit-btn');
    await expect(submitButton).toBeDisabled();
  });

  test('should navigate to Week 1 challenge page', async ({ page }) => {
    // Click on Week 1 Start Challenge link
    await page.locator('a[href="week1.html"]').first().click();

    // Verify navigation
    await expect(page).toHaveURL(/week1/);
    await expect(page.locator('h1')).toContainText('UIGen Feature Sprint');
  });
});

test.describe('Week 1 Challenge Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/week1.html');
  });

  test('should display challenge information', async ({ page }) => {
    // Title
    await expect(page.locator('h1')).toContainText('UIGen Feature Sprint');

    // Challenge details
    await expect(page.locator('text=70 min')).toBeVisible();
    await expect(page.locator('text=3-Stage Sprint')).toBeVisible();
    await expect(page.locator('text=100 +20')).toBeVisible();
  });

  test('should display all three stages', async ({ page }) => {
    // Stage 1: Clear All Files
    await expect(page.locator('h3').filter({ hasText: 'Clear All Files' })).toBeVisible();

    // Stage 2: Download as ZIP
    await expect(page.locator('h3').filter({ hasText: 'Download as ZIP' })).toBeVisible();

    // Stage 3: Keyboard Shortcuts
    await expect(page.locator('h3').filter({ hasText: 'Keyboard Shortcuts' })).toBeVisible();
  });

  test('should display scoring rubric', async ({ page }) => {
    await expect(page.locator('h2').filter({ hasText: 'SCORING RUBRIC' })).toBeVisible();

    // Check point values
    await expect(page.locator('text=25').first()).toBeVisible(); // Stage 1
    await expect(page.locator('text=30').first()).toBeVisible(); // Stage 2
  });

  test('should have download project link', async ({ page }) => {
    const downloadLink = page.locator('a[href*="uigen.zip"]');
    await expect(downloadLink).toBeVisible();
  });

  test('should show timer section', async ({ page }) => {
    // Timer/Submit section exists
    await expect(page.locator('h2').filter({ hasText: 'SUBMIT YOUR WORK' })).toBeVisible();

    // Start Timer button (may be disabled if challenge not started)
    const startTimerButton = page.locator('button').filter({ hasText: 'Start Timer' });
    await expect(startTimerButton).toBeVisible();
  });
});

test.describe('Leaderboard Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/leaderboard.html');
  });

  test('should display leaderboard page', async ({ page }) => {
    await expect(page.locator('h1')).toContainText('Leaderboard');
  });

  test('should have season and week tabs', async ({ page }) => {
    await expect(page.locator('button').filter({ hasText: 'Season' })).toBeVisible();
    await expect(page.locator('button').filter({ hasText: 'Week 1' })).toBeVisible();
  });
});

test.describe('Authenticated User Flow', () => {
  // These tests require valid test credentials
  // Set TEST_USER_ID and TEST_PASSWORD environment variables

  test.skip(
    !process.env.TEST_USER_ID || !process.env.TEST_PASSWORD,
    'Skipping auth tests - TEST_USER_ID and TEST_PASSWORD not set'
  );

  test('should login successfully with valid credentials', async ({ page }) => {
    await page.goto('/');

    // Open login modal
    await page.locator('#auth-buttons button').filter({ hasText: 'Login' }).click();

    // Fill credentials
    await page.locator('#login-userid').fill(process.env.TEST_USER_ID!);
    await page.locator('#login-password').fill(process.env.TEST_PASSWORD!);

    // Submit
    await page.locator('#login-form button[type="submit"]').click();

    // Wait for auth to complete
    await page.waitForSelector('#user-profile:not(.hidden)', { timeout: 5000 });

    // Verify logged in
    await expect(page.locator('#user-profile')).toBeVisible();
    await expect(page.locator('#auth-buttons')).toHaveClass(/hidden/);
  });

  test('should be able to submit after login', async ({ page }) => {
    // Login first
    await page.goto('/');
    await page.locator('#auth-buttons button').filter({ hasText: 'Login' }).click();
    await page.locator('#login-userid').fill(process.env.TEST_USER_ID!);
    await page.locator('#login-password').fill(process.env.TEST_PASSWORD!);
    await page.locator('#login-form button[type="submit"]').click();
    await page.waitForSelector('#user-profile:not(.hidden)', { timeout: 5000 });

    // Check submit section - login warning should be hidden
    await expect(page.locator('#login-required-warning')).toHaveClass(/hidden/);
  });

  test('should logout successfully', async ({ page }) => {
    // Login first
    await page.goto('/');
    await page.locator('#auth-buttons button').filter({ hasText: 'Login' }).click();
    await page.locator('#login-userid').fill(process.env.TEST_USER_ID!);
    await page.locator('#login-password').fill(process.env.TEST_PASSWORD!);
    await page.locator('#login-form button[type="submit"]').click();
    await page.waitForSelector('#user-profile:not(.hidden)', { timeout: 5000 });

    // Click logout
    await page.locator('button').filter({ hasText: 'Logout' }).click();

    // Verify logged out
    await expect(page.locator('#auth-buttons')).not.toHaveClass(/hidden/);
    await expect(page.locator('#user-profile')).toHaveClass(/hidden/);
  });
});
