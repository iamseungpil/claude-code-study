# Week 1 Playwright Tests

Playwright E2E tests for Week 1 UIGen Challenge evaluation.

## Overview

This directory contains two types of tests:

1. **Site Tests** (`site.spec.ts`) - Tests for the deployed Claude Code Study site
   - Login/Signup flow
   - Challenge participation flow
   - Main page elements

2. **UIGen Evaluation Tests** (`uigen.spec.ts`) - Tests for evaluating submitted UIGen projects
   - Stage 1: Clear All Files (20 points)
   - Stage 2: Download ZIP (25 points)
   - Stage 3: Keyboard Shortcuts (25 points)

## Setup

```bash
# Install dependencies
cd tests/week1
npm install

# Install Playwright browsers
npx playwright install
```

## Running Tests

### Site Tests

Test the deployed study site:

```bash
# Run all site tests
npx playwright test --project=site-tests

# Run with UI mode
npx playwright test --project=site-tests --ui

# Run with custom URL
SITE_URL=http://localhost:8003 npx playwright test --project=site-tests
```

### UIGen Evaluation Tests

Test a submitted UIGen project:

```bash
# 1. Start the submitted project's dev server
cd /path/to/submitted/project
npm run dev  # Usually runs on localhost:3000

# 2. Run evaluation tests
cd tests/week1
npx playwright test --project=uigen-evaluation

# Run with custom URL
UIGEN_URL=http://localhost:3001 npx playwright test --project=uigen-evaluation
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SITE_URL` | `https://claude-code-study.pages.dev/frontend/` | URL of the deployed study site |
| `UIGEN_URL` | `http://localhost:3000` | URL of the UIGen project to evaluate |
| `TEST_USER_ID` | - | Test user ID for authentication tests |
| `TEST_PASSWORD` | - | Test password for authentication tests |

## Scoring System

### Playwright-Testable Points (70/100)

| Test | Points | Description |
|------|--------|-------------|
| Stage 1: Clear button exists | 10 | Button with Trash2 icon or "Clear" text |
| Stage 1: Dialog works | 10 | Confirmation dialog shows and Cancel works |
| Stage 2: Download button exists | 10 | Button with Download icon or "Download" text |
| Stage 2: ZIP downloads | 15 | Clicking triggers .zip file download |
| Stage 3: Cmd+K opens palette | 10 | Command palette opens on keyboard shortcut |
| Stage 3: Has commands | 10 | Palette contains Clear and Download commands |
| Stage 3: ESC closes | 5 | Pressing Escape closes the palette |

### Manual/Claude Evaluation (30/100)

| Category | Points | Description |
|----------|--------|-------------|
| Memory Record (Stage 1) | 5 | CLAUDE.md documents Stage 1 learnings |
| Memory Record (Stage 2) | 5 | CLAUDE.md documents Stage 2 learnings |
| CLAUDE.md Quality | 10 | Overall quality of documentation |
| Code Quality | 10 | TypeScript usage, patterns, error handling |

## Test Results

After running tests, results are saved to:

- `test-results/results.json` - Full Playwright JSON report
- `test-results/score-report.json` - Score breakdown for evaluation

## Using with the Study Platform

These Playwright tests are optional developer tooling. The current study platform uses **manual admin review** for scoring.

If you want to run the tests locally for debugging a submission, run them directly from `tests/week1/`.

## Troubleshooting

### Browser not installed

```bash
npx playwright install chromium
```

### Dev server not starting

Make sure port 3000 is free:
```bash
lsof -i :3000
```

### Tests timing out

Increase timeout in playwright.config.ts or use:
```bash
npx playwright test --timeout=60000
```
