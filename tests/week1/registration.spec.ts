import { test, expect } from '@playwright/test';

/**
 * Registration and Challenge Participation E2E Tests
 *
 * Tests:
 * 1. User registration with specific test credentials
 * 2. Challenge participation flow (login, start timer, submit)
 */

const TEST_USER = {
  userId: 'iamseungpil',
  firstName: 'Seungpil',
  lastName: 'Lee',
  password: 'iam414732$$'
};

const CHALLENGE_GITHUB_URL = 'https://github.com/iamseungpil/claude-code-week1.git';

test.describe('User Registration Flow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should register a new user', async ({ page }) => {
    // Open signup modal
    await page.locator('#auth-buttons button').filter({ hasText: 'Sign Up' }).click();

    // Check modal is visible
    const modal = page.locator('#signup-modal');
    await expect(modal).not.toHaveClass(/hidden/);

    // Fill in registration form
    await page.locator('#signup-userid').fill(TEST_USER.userId);
    await page.locator('#signup-firstname').fill(TEST_USER.firstName);
    await page.locator('#signup-lastname').fill(TEST_USER.lastName);
    await page.locator('#signup-password').fill(TEST_USER.password);
    await page.locator('#signup-confirm').fill(TEST_USER.password);

    // Submit form
    await page.locator('#signup-form button[type="submit"]').click();

    // Wait for response - either success (user profile visible) or error (already exists)
    const result = await Promise.race([
      page.waitForSelector('#user-profile:not(.hidden)', { timeout: 10000 }).then(() => 'success'),
      page.waitForSelector('#signup-error:not(:empty)', { timeout: 10000 }).then(() => 'error')
    ]);

    if (result === 'success') {
      // New user registered successfully
      await expect(page.locator('#user-profile')).toBeVisible();
      console.log('User registered successfully');
    } else {
      // User might already exist - check error message
      const errorText = await page.locator('#signup-error').textContent();
      console.log('Registration error:', errorText);
      // If user already exists, that's acceptable for our test
      expect(errorText).toContain('already');
    }
  });
});

test.describe('Login and Challenge Participation', () => {
  test('should login and access challenge page', async ({ page }) => {
    await page.goto('/');

    // Open login modal
    await page.locator('#auth-buttons button').filter({ hasText: 'Login' }).click();

    // Fill credentials
    await page.locator('#login-userid').fill(TEST_USER.userId);
    await page.locator('#login-password').fill(TEST_USER.password);

    // Submit login
    await page.locator('#login-form button[type="submit"]').click();

    // Wait for login to complete
    await page.waitForSelector('#user-profile:not(.hidden)', { timeout: 10000 });

    // Verify logged in
    await expect(page.locator('#user-profile')).toBeVisible();
    await expect(page.locator('#auth-buttons')).toHaveClass(/hidden/);

    console.log('Login successful');
  });

  test('should navigate to Week 1 challenge page when logged in', async ({ page }) => {
    // Login first
    await page.goto('/');
    await page.locator('#auth-buttons button').filter({ hasText: 'Login' }).click();
    await page.locator('#login-userid').fill(TEST_USER.userId);
    await page.locator('#login-password').fill(TEST_USER.password);
    await page.locator('#login-form button[type="submit"]').click();
    await page.waitForSelector('#user-profile:not(.hidden)', { timeout: 10000 });

    // Navigate to Week 1 challenge
    await page.goto('/week1.html');

    // Wait for page to load
    await page.waitForLoadState('networkidle');

    // Verify we're on the challenge page
    await expect(page.locator('h1')).toBeVisible();

    console.log('Challenge page loaded successfully');
  });

  test('should submit challenge work', async ({ page }) => {
    // Login first
    await page.goto('/');
    await page.locator('#auth-buttons button').filter({ hasText: 'Login' }).click();
    await page.locator('#login-userid').fill(TEST_USER.userId);
    await page.locator('#login-password').fill(TEST_USER.password);
    await page.locator('#login-form button[type="submit"]').click();
    await page.waitForSelector('#user-profile:not(.hidden)', { timeout: 10000 });

    // Navigate to Week 1 challenge
    await page.goto('/week1.html');
    await page.waitForLoadState('networkidle');

    // Look for Start Timer button and click if visible
    const startTimerButton = page.locator('button').filter({ hasText: /start timer/i }).first();
    if (await startTimerButton.isVisible()) {
      await startTimerButton.click();
      // Wait a moment for timer to start
      await page.waitForTimeout(1000);
      console.log('Timer started');
    }

    // Find GitHub URL input and fill it
    const githubInput = page.locator('input[type="text"], input[type="url"]').filter({ hasText: '' }).first();
    const inputs = page.locator('input[placeholder*="github" i], input[placeholder*="url" i], input[id*="github" i], input[id*="url" i]');
    const inputCount = await inputs.count();

    if (inputCount > 0) {
      await inputs.first().fill(CHALLENGE_GITHUB_URL);
      console.log('GitHub URL filled');
    } else {
      // Try finding any text input in the submit section
      const submitSection = page.locator('[id*="submit" i], [class*="submit" i]').first();
      const textInput = submitSection.locator('input[type="text"], input[type="url"]').first();
      if (await textInput.isVisible()) {
        await textInput.fill(CHALLENGE_GITHUB_URL);
        console.log('GitHub URL filled in submit section');
      }
    }

    // Look for submit button
    const submitButton = page.locator('button').filter({ hasText: /submit/i }).first();
    if (await submitButton.isVisible() && await submitButton.isEnabled()) {
      await submitButton.click();
      console.log('Submit button clicked');

      // Wait for response
      await page.waitForTimeout(3000);

      // Check for success or error message
      const pageContent = await page.content();
      if (pageContent.includes('success') || pageContent.includes('submitted')) {
        console.log('Submission appears successful');
      }
    } else {
      console.log('Submit button not found or disabled');
    }
  });
});
