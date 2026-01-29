# Claude Code Hooks

This directory contains custom hooks for the Claude Code Study project.

## Available Hooks

### `update-webhook.sh`

**Purpose**: Reminds you to update the GitHub webhook URL when the Cloudflare Tunnel URL changes.

**When to use**:
- After restarting the cloudflared tunnel
- When you see connection errors from GitHub webhook
- Before pushing critical updates that need auto-deployment

**Usage**:
```bash
./.claude/hooks/update-webhook.sh
```

**What it does**:
1. Reads the current Cloudflare Tunnel URL from `cloudflared.log`
2. Compares it with the URL configured in `frontend/config.js`
3. Displays step-by-step instructions for updating the GitHub webhook

## Why Manual Updates Are Needed

**Cloudflare Quick Tunnels** generate random URLs each time they start:
- `https://random-words-here.trycloudflare.com`
- These URLs are different every time cloudflared restarts

**GitHub Webhooks** need a stable URL to send push notifications to your backend.

When the tunnel URL changes, the webhook still points to the old URL, causing:
- ❌ Auto-deployment failures
- ❌ Push events not triggering backend updates
- ❌ "Connection refused" errors in GitHub webhook logs

## Solution Options

### Option 1: Manual Update (Current)
Run `update-webhook.sh` after each tunnel restart and follow the instructions.

### Option 2: Named Tunnel (Recommended for Production)
Create a persistent Cloudflare Tunnel with a fixed URL:

```bash
# Install cloudflared
# Create a named tunnel
cloudflared tunnel create claude-study

# Configure the tunnel
# The URL will be: https://claude-study.yourdomain.com
# This URL never changes, even after restarts
```

See: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/

### Option 3: ngrok or Similar
Use a service that provides stable URLs for local development.

## Integration with Claude Code

You can run this hook automatically by:

1. **Adding to your workflow**: Run it before important pushes
2. **Setting up a reminder**: Create a task in your task tracker
3. **Automating checks**: Add it to your CI/CD pipeline

## Troubleshooting

**Q: Hook says "Tunnel URL not found"**
- Check if cloudflared is running: `ps aux | grep cloudflared`
- Check if cloudflared.log exists and has recent entries

**Q: How do I know if webhook is working?**
- Go to GitHub → Settings → Webhooks
- Click on your webhook
- Check "Recent Deliveries" tab
- Look for green checkmarks (200 OK responses)

**Q: Webhook shows 403 or 500 errors**
- 403: Check webhook secret matches `.env.webhook`
- 500: Check backend logs: `tail -f backend.log`
