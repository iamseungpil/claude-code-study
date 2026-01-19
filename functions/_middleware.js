// Cloudflare Pages Middleware
// Directly serve HTML files using ASSETS to bypass SPA fallback

export async function onRequest(context) {
  const { request, env, next } = context;
  const url = new URL(request.url);
  const path = url.pathname;

  // List of HTML files that should be served directly
  const htmlFiles = [
    '/frontend/admin.html',
    '/frontend/leaderboard.html',
    '/frontend/week1.html',
    '/frontend/week2.html',
    '/frontend/week3.html',
    '/frontend/week4.html',
    '/frontend/week5.html',
    '/frontend/week1-learn.html',
    '/frontend/week2-learn.html',
    '/frontend/week3-learn.html',
    '/frontend/week4-learn.html',
    '/frontend/week5-learn.html',
    '/frontend/index.html',
  ];

  // For HTML files, fetch directly from ASSETS
  if (htmlFiles.includes(path) || (path.startsWith('/frontend/') && path.endsWith('.html'))) {
    try {
      // Create a new request for the asset
      const assetRequest = new Request(url.toString(), {
        method: 'GET',
        headers: request.headers,
      });

      // Fetch from ASSETS binding
      const response = await env.ASSETS.fetch(assetRequest);

      // If we got a valid response with correct content type, return it
      if (response.ok) {
        // Clone response and add cache headers
        return new Response(response.body, {
          status: response.status,
          headers: {
            ...Object.fromEntries(response.headers),
            'Cache-Control': 'no-cache',
          },
        });
      }
    } catch (e) {
      console.error('Error fetching asset:', e);
    }
  }

  // For /frontend/ without file, redirect to index.html
  if (path === '/frontend/' || path === '/frontend') {
    return Response.redirect(url.origin + '/frontend/index.html', 302);
  }

  // For JS/CSS files, let them pass through
  if (path.endsWith('.js') || path.endsWith('.css')) {
    return next();
  }

  // Default: pass through
  return next();
}
