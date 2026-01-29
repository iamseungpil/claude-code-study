#!/usr/bin/env bash
# Webhook Update Reminder Hook
#
# This script reminds you to update the GitHub webhook URL when the Cloudflare Tunnel changes.
#
# IMPORTANT: Quick Tunnels generate random URLs on each restart.
# The GitHub webhook must be manually updated whenever the tunnel URL changes.

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔔 WEBHOOK UPDATE REMINDER"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Get current tunnel URL from log
TUNNEL_URL=$(grep "trycloudflare.com" cloudflared.log | grep -o "https://[^[:space:]]*trycloudflare.com" | tail -1 || echo "NOT FOUND")

if [ "$TUNNEL_URL" = "NOT FOUND" ]; then
    echo "❌ Cloudflare Tunnel URL not found in cloudflared.log"
    echo ""
    echo "Please check if cloudflared is running:"
    echo "  ./cloudflared-bin/cloudflared.exe tunnel --url http://localhost:8003 --protocol http2"
    exit 1
fi

# Get current config URL
CONFIG_URL=$(grep "CONFIGURED_API_BASE" frontend/config.js | grep -o "https://[^'\"]*trycloudflare.com" || echo "NOT SET")

echo "📍 Current Tunnel URL:"
echo "   $TUNNEL_URL"
echo ""
echo "📍 Config.js API URL:"
echo "   $CONFIG_URL"
echo ""

# Check if they match
if [ "$TUNNEL_URL" != "$CONFIG_URL" ]; then
    echo "⚠️  WARNING: URLs do not match!"
    echo ""
    echo "Update frontend/config.js:"
    echo "   CONFIGURED_API_BASE = '$TUNNEL_URL'"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 MANUAL ACTION REQUIRED:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Go to: https://github.com/iamseungpil/claude-code-study/settings/hooks"
echo ""
echo "2. Click on the existing webhook (or 'Add webhook')"
echo ""
echo "3. Update the Payload URL to:"
echo "   $TUNNEL_URL/webhook/github"
echo ""
echo "4. Ensure these settings:"
echo "   - Content type: application/json"
echo "   - Secret: (already configured in .env.webhook)"
echo "   - Events: Just the push event"
echo "   - Active: ✓"
echo ""
echo "5. Click 'Update webhook'"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💡 TIP: Save this for future reference"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Tunnel URLs change every time cloudflared restarts."
echo "You must update the webhook URL each time."
echo ""
echo "To avoid this, consider using a Named Tunnel:"
echo "  https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/tunnel-guide/"
echo ""
