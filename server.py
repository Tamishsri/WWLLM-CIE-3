#!/usr/bin/env python3
"""
server.py — WWLLM Lecture Studio
================================
Serves the web studio (``web/``) and its streaming API (``src/api``).

    python server.py                  # http://127.0.0.1:8000
    python server.py --open           # …and open it in the browser
    python server.py --demo           # force the offline engines
    python server.py --offline        # models from the local cache only, no network
    python server.py --host 0.0.0.0   # expose on your network

Each stage uses a stored result, else the real model, else an offline engine,
so the studio runs on any laptop.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import webbrowser


def free_port(host: str, port: int) -> int:
    """Return *port*, or the next free one if something is already using it."""
    for candidate in range(port, port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if probe.connect_ex(("127.0.0.1" if host in ("0.0.0.0", "::") else host, candidate)) != 0:
                return candidate
    return port


def main() -> None:
    parser = argparse.ArgumentParser(description="WWLLM Lecture Studio server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default 8000)")
    parser.add_argument("--demo", action="store_true", help="Force offline demo engines")
    parser.add_argument("--offline", action="store_true", help="Load models from the local cache only")
    parser.add_argument("--open", action="store_true", help="Open the studio in a browser")
    parser.add_argument(
        "--pace",
        type=float,
        default=1.0,
        help="Speed of offline-stage animations (0 = instant, default 1)",
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.demo:
        os.environ["WWLLM_DEMO"] = "1"
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import uvicorn

    from src.api.app import create_app

    port = free_port(args.host, args.port)
    if port != args.port:
        print(f"\n  Port {args.port} is busy, using {port} instead.")
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{port}"
    print(f"\n  WWLLM Lecture Studio  ->  {url}\n")
    if args.open:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    uvicorn.run(create_app(demo_pace=args.pace), host=args.host, port=port, log_level="info")


if __name__ == "__main__":
    main()
