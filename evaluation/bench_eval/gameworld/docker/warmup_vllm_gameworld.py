#!/usr/bin/env python3
"""Warm a vLLM endpoint with the GameWorld 480p/history-5 request shape."""

from __future__ import annotations

import argparse
import base64
import io

import requests
from PIL import Image

ACTION = "<|action_start|>0 0 0 ; ; ; ; ; ;<|action_end|>"


def image_url(width: int, height: int) -> str:
    image = Image.new("RGB", (width, height), color=(32, 32, 32))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_messages(url: str, history: int) -> list[dict]:
    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are an experienced PC game player. Output exactly one "
                "<|action_start|>...<|action_end|> action."
            ),
        }
    ]
    for _ in range(history):
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "current game screen"},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        )
        messages.append({"role": "assistant", "content": ACTION})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "current game screen"},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }
    )
    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True, help="Base URL, e.g. http://127.0.0.1:8088")
    parser.add_argument("--model", default="/model")
    parser.add_argument("--width", type=int, default=853)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    response = requests.post(
        f"{args.endpoint.rstrip('/')}/v1/chat/completions",
        json={
            "model": args.model,
            "messages": build_messages(image_url(args.width, args.height), args.history),
            "temperature": 0.0,
            "max_tokens": 8,
        },
        timeout=args.timeout,
    )
    response.raise_for_status()
    payload = response.json()
    usage = payload.get("usage") or {}
    print(
        f"[WARMUP] endpoint={args.endpoint} status={response.status_code} "
        f"prompt_tokens={usage.get('prompt_tokens')} completion_tokens={usage.get('completion_tokens')}"
    )


if __name__ == "__main__":
    main()
