#!/usr/bin/env bash
# Fake OpenAI-compatible server for testing managed_server lifecycle.
# Writes /tmp/fake_serve_started on launch, serves /health -> 200, exits on SIGTERM.
set -euo pipefail

PORT="${FAKE_SERVE_PORT:-8765}"
echo "started" > /tmp/fake_serve_started

exec python3 - "$PORT" <<'PYEOF'
import http.server
import sys

port = int(sys.argv[1])


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


http.server.HTTPServer(("127.0.0.1", port), Handler).serve_forever()
PYEOF