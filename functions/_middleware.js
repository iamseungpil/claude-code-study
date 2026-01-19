// Cloudflare Pages Middleware
// Disable SPA fallback - serve actual HTML files

export async function onRequest(context) {
  const { request, next, env } = context;
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

  // If requesting an HTML file, let it pass through to static assets
  if (htmlFiles.includes(path) || path.endsWith('.html') || path.endsWith('.js') || path.endsWith('.css')) {
    return next();
  }

  // For /frontend/ without file, serve index.html
  if (path === '/frontend/' || path === '/frontend') {
    return next();
  }

  // Default: pass through to static assets
  return next();
}
