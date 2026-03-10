"""
Project Alpha — Microservice Gateway

Entry point for the API gateway service.
"""

import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROUTES = {
    "/api/users": "http://user-service:8001",
    "/api/orders": "http://order-service:8002",
    "/api/products": "http://product-service:8003",
}


class GatewayHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "healthy"})
            return

        backend = self._match_route(self.path)
        if backend:
            logger.info(f"Routing {self.path} -> {backend}")
            self._respond(200, {"routed_to": backend, "path": self.path})
        else:
            self._respond(404, {"error": "No route matched"})

    def _match_route(self, path: str) -> str | None:
        for pattern, backend in ROUTES.items():
            if path.startswith(pattern):
                return backend
        return None

    def _respond(self, status: int, body: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())


def main():
    server = HTTPServer(("0.0.0.0", 8080), GatewayHandler)
    logger.info("Gateway listening on :8080")
    server.serve_forever()


if __name__ == "__main__":
    main()
