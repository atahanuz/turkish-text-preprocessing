const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Proxy API requests to Flask backend
// When using app.use('/api', ...), the middleware receives path WITHOUT /api
// So we need to prepend /api back when forwarding to Flask
app.use('/api', createProxyMiddleware({
  target: 'http://localhost:5001',
  changeOrigin: true,
  pathRewrite: (path, req) => {
    const newPath = '/api' + path; // Prepend /api back
    console.log(`Proxying ${req.method} ${path} -> http://localhost:5001${newPath}`);
    return newPath;
  },
  onError: (err, req, res) => {
    console.error('Proxy error:', err);
    res.status(500).json({
      error: 'Backend connection failed. Make sure Flask server is running on port 5001.'
    });
  }
}));

// Serve static files from the React app build directory
app.use(express.static(path.join(__dirname, 'build')));

// Handle all other routes by serving index.html (for React Router)
// Using middleware instead of route pattern to avoid Express 5 path-to-regexp issues
app.use((req, res) => {
  res.sendFile(path.join(__dirname, 'build', 'index.html'));
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`
┌─────────────────────────────────────────────────────┐
│  🚀 Server running on http://localhost:${PORT}       │
│  📡 Proxying /api/* to http://localhost:5001       │
│  📦 Serving React app from /build                   │
│                                                     │
│  Make sure Flask backend is running:               │
│  python server.py                                  │
└─────────────────────────────────────────────────────┘
  `);
});
