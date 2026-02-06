#!/usr/bin/env python
# run_api.py
# Запуск API сервера

import argparse
import sys
import os

# Добавляем src в path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def main():
    parser = argparse.ArgumentParser(description="Speed Limit API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════╗
║           Speed Limit API Server                         ║
╠══════════════════════════════════════════════════════════╣
║  REST API:    http://{args.host}:{args.port}/api
║  WebSocket:   ws://{args.host}:{args.port}/ws
║  Swagger UI:  http://{args.host}:{args.port}/docs
║  ReDoc:       http://{args.host}:{args.port}/redoc
╚══════════════════════════════════════════════════════════╝
    """)

    import uvicorn

    if args.reload:
        # Режим разработки с auto-reload
        uvicorn.run(
            "src.api.server:app",
            host=args.host,
            port=args.port,
            reload=True,
            reload_dirs=["src"],
        )
    else:
        # Production режим
        from src.api.server import app
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
