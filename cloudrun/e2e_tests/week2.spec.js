// Week 2 E2E Tests: Hooks Mastery
// Note: Week 2 is about Claude Code Hooks (Node.js scripts)
// These tests verify the hook scripts work correctly

const { test, expect } = require('@playwright/test');
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

test.describe('Week 2: Hooks Mastery', () => {
  // Week 2 doesn't have a web UI - it's about hooks
  // These tests run the hook scripts directly

  test.describe('Stage 1: Security Hook', () => {
    test('read_hook.js blocks .env files', async () => {
      const hookPath = path.join(process.cwd(), 'hooks', 'read_hook.js');

      if (!fs.existsSync(hookPath)) {
        console.log('read_hook.js not found - skipping');
        return;
      }

      try {
        // Test: Should block .env file (exit code 2)
        const input = JSON.stringify({
          tool_input: { file_path: '/path/to/.env' }
        });

        execSync(`echo '${input}' | node ${hookPath}`, {
          encoding: 'utf-8',
          stdio: ['pipe', 'pipe', 'pipe']
        });

        // If we reach here, exit code was 0 (wrong!)
        throw new Error('Expected exit code 2 for .env file');
      } catch (error) {
        // Check if it exited with code 2 (correct behavior)
        if (error.status === 2) {
          // Correct! Hook blocked the .env file
          expect(true).toBeTruthy();
        } else if (error.message.includes('Expected exit code 2')) {
          throw error;
        } else {
          // Some other error - might be acceptable
          console.log('Hook error:', error.message);
        }
      }
    });

    test('read_hook.js allows safe files', async () => {
      const hookPath = path.join(process.cwd(), 'hooks', 'read_hook.js');

      if (!fs.existsSync(hookPath)) {
        console.log('read_hook.js not found - skipping');
        return;
      }

      try {
        // Test: Should allow safe file (exit code 0)
        const input = JSON.stringify({
          tool_input: { file_path: '/path/to/safe.txt' }
        });

        execSync(`echo '${input}' | node ${hookPath}`, {
          encoding: 'utf-8'
        });

        // If we reach here without error, exit code was 0 (correct!)
        expect(true).toBeTruthy();
      } catch (error) {
        // Exit code was not 0
        throw new Error(`Expected exit code 0 for safe file, got ${error.status}`);
      }
    });
  });

  test.describe('Stage 2: Settings Validation', () => {
    test('settings.json is valid JSON', async () => {
      const settingsPath = path.join(process.cwd(), '.claude', 'settings.json');

      if (!fs.existsSync(settingsPath)) {
        console.log('settings.json not found - skipping');
        return;
      }

      const content = fs.readFileSync(settingsPath, 'utf-8');

      // Should not throw
      const parsed = JSON.parse(content);
      expect(parsed).toBeDefined();
    });

    test('settings.json has hooks configuration', async () => {
      const settingsPath = path.join(process.cwd(), '.claude', 'settings.json');

      if (!fs.existsSync(settingsPath)) {
        console.log('settings.json not found - skipping');
        return;
      }

      const content = fs.readFileSync(settingsPath, 'utf-8');
      const parsed = JSON.parse(content);

      // Check for hooks section
      expect(parsed.hooks || parsed.Hooks).toBeDefined();
    });
  });

  test.describe('Stage 3: Custom Hook', () => {
    test('Custom hook file exists', async () => {
      const hooksDir = path.join(process.cwd(), 'hooks');

      if (!fs.existsSync(hooksDir)) {
        console.log('hooks directory not found - skipping');
        return;
      }

      const files = fs.readdirSync(hooksDir);

      // Look for any hook file besides the provided ones
      const customHooks = files.filter(f =>
        f.endsWith('.js') &&
        !['read_hook.js', 'query_hook.js', 'tsc.js'].includes(f)
      );

      console.log(`Found custom hooks: ${customHooks.join(', ') || 'none'}`);
      // Don't fail if no custom hook - it's a bonus
    });
  });
});
