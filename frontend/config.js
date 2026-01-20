/**
 * Claude Code Study - Frontend Configuration
 *
 * This file centrally manages frontend environment settings.
 * 
 * DEPLOYMENT MODES:
 * 1. Local Development: Both frontend & backend on localhost (auto-detected)
 * 2. Cloudflare Pages + Local Backend: Set CONFIGURED_API_BASE to Cloudflare Tunnel URL
 * 3. Full Cloud: Set CONFIGURED_API_BASE to your cloud backend URL
 * 
 * To expose local backend via Cloudflare Tunnel:
 *   $ cloudflared tunnel --url http://localhost:8003
 *   Then copy the generated URL (e.g., https://xxx.trycloudflare.com)
 */

(function() {
    'use strict';

    // ============================================================
    // Production Settings
    // ============================================================

    // Backend API URL (via Cloudflare Tunnel)
    // Set to your Cloudflare Tunnel URL when exposing local backend
    // Leave empty for local development (auto-detected)
    const CONFIGURED_API_BASE = 'https://measuring-strikes-clothing-stewart.trycloudflare.com';

    // ============================================================
    // API Base URL Detection Logic
    // ============================================================

    const STORAGE_KEY = 'claude_code_study_api_base';

    const getApiBase = () => {
        // 1. URL parameter override (for testing): ?api=https://xxx.trycloudflare.com
        // Also save to localStorage for persistence across page navigations
        const urlParams = new URLSearchParams(window.location.search);
        const apiParam = urlParams.get('api');
        if (apiParam) {
            console.log('[Config] API_BASE from URL param:', apiParam);
            try {
                localStorage.setItem(STORAGE_KEY, apiParam);
            } catch (e) {
                console.warn('[Config] Could not save API_BASE to localStorage:', e);
            }
            return apiParam;
        }

        // 2. Check localStorage for previously saved API_BASE (from URL param)
        try {
            const savedApiBase = localStorage.getItem(STORAGE_KEY);
            if (savedApiBase) {
                console.log('[Config] API_BASE from localStorage:', savedApiBase);
                return savedApiBase;
            }
        } catch (e) {
            console.warn('[Config] Could not read API_BASE from localStorage:', e);
        }

        // 3. Use explicitly configured value if available
        if (CONFIGURED_API_BASE) {
            return CONFIGURED_API_BASE;
        }

        // 4. Local development: FastAPI serving on same port (localhost:8003)
        if (window.location.port === '8003') {
            return '';  // same-origin
        }

        // 5. Local development: Frontend served on different port (e.g., VS Code Live Server)
        if (window.location.hostname === 'localhost' ||
            window.location.hostname === '127.0.0.1') {
            return 'http://localhost:8003';
        }

        // 6. Production (Cloudflare Pages): API not configured
        console.warn(
            '[Config] API_BASE not configured for production.\n' +
            'Options:\n' +
            '  1. Edit config.js and set CONFIGURED_API_BASE\n' +
            '  2. Add ?api=YOUR_BACKEND_URL to the URL'
        );
        return '';
    };

    // ============================================================
    // Export Global Settings
    // ============================================================

    // API Base URL
    window.API_BASE = getApiBase();

    // Backward compatibility: also set __API_BASE__
    window.__API_BASE__ = window.API_BASE;

    // Debug log (development environment only)
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
        console.log('[Config] API_BASE:', window.API_BASE || '(same-origin)');
    }

})();
