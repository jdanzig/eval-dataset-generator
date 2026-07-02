"""Anthropic Messages client. Drops `temperature` and retries on models that reject it."""
from __future__ import annotations
import os
URL = "https://api.anthropic.com/v1/messages"

def complete(model: str, user: str, system: str | None = None, max_tokens: int = 512,
             temperature: float | None = 0.0) -> str:
    import httpx
    body: dict = {"model": model, "max_tokens": max_tokens,
                  "messages": [{"role": "user", "content": user}]}
    if system: body["system"] = system
    if temperature is not None: body["temperature"] = temperature
    headers = {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"}
    for _ in range(2):
        r = httpx.post(URL, headers=headers, json=body, timeout=90)
        if r.status_code == 400 and "temperature" in r.text and "deprecated" in r.text:
            body.pop("temperature", None); continue
        r.raise_for_status()
        return "".join(b.get("text","") for b in r.json()["content"] if b.get("type")=="text").strip()
    r.raise_for_status(); return ""
