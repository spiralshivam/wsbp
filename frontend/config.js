// Same-origin API when Flask serves the app. For the optional separate local
// frontend server (port 8000), use Flask's default port 5000.
// Set this to the backend URL when deploying the frontend separately.
window.API_BASE = window.API_BASE || (window.location.port === "8000"
  ? "http://localhost:5000"
  : window.location.origin);
