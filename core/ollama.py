"""Ollama plumbing: URL detection, model resolution, and an injectable client
whose chat() does structured-output streaming with live thinking and a deadline
cutoff. Tests substitute a MockClient with the same chat() signature."""

import json
import os
import re
import sys
import time

import requests

from core.term import DIM, RESET


def _is_wsl():
    """True only when running under WSL (where localhost can't reach Windows)."""
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def detect_ollama_url():
    """Find Ollama by PROBING candidates — guessing from network config breaks
    whenever WSL switches between NAT and mirrored networking (a stale
    OLLAMA_HOST, or a "gateway" that is really the LAN router, would win)."""
    cands = candidate_urls()
    for u in cands:
        try:
            requests.get(f"{u}/api/tags", timeout=2).raise_for_status()
            return u
        except Exception:
            continue
    return cands[0]  # nothing answered; list_installed() will explain clearly


def candidate_urls():
    """Possible Ollama endpoints, most-specific first: OLLAMA_URL/OLLAMA_HOST
    env vars, localhost (correct under WSL mirrored networking), then the WSL2
    gateway IP (correct under WSL NAT networking)."""
    urls = []
    for var in ("OLLAMA_URL", "OLLAMA_HOST"):
        v = os.environ.get(var, "").strip().rstrip("/")
        if v:
            if not v.startswith("http"):
                v = f"http://{v}"
            if not re.search(r":\d+$", v):
                v = f"{v}:11434"
            urls.append(v)
    urls.append("http://localhost:11434")
    if _is_wsl():
        try:
            import subprocess
            out = subprocess.check_output(["ip", "route", "show", "default"],
                                          text=True)
            m = re.search(r"via\s+(\d+\.\d+\.\d+\.\d+)", out)
            if m:
                urls.append(f"http://{m.group(1)}:11434")
        except Exception:
            pass
    return list(dict.fromkeys(urls))


def resolve_model(requested, explicit, installed):
    """Resolve a model name against installed models. Exact tag wins; a bare
    name matches '<name>:latest' or the first '<name>:*'. An explicit miss
    errors; a default miss falls back to the largest installed model."""
    names = [n for n, _ in installed]
    if requested in names:
        return requested
    if ":" not in requested:
        if f"{requested}:latest" in names:
            return f"{requested}:latest"
        family = [n for n in names if n.split(":")[0] == requested]
        if family:
            return family[0]

    if not installed:
        print("No models installed in Ollama. Pull one, e.g. 'ollama pull gemma3:12b'.")
        sys.exit(1)
    if explicit:
        print(f"Model '{requested}' not found. Available: {', '.join(names)}")
        sys.exit(1)

    largest = installed[0][0]
    print(f"{DIM}Default model '{requested}' not installed; "
          f"using largest available: {largest}{RESET}")
    return largest


class OllamaClient:
    """Thin client over Ollama's /api. The engine only calls list_installed()
    and chat(); anything with the same surface (see tests/mockclient.py) works."""

    def __init__(self, base_url):
        self.base_url = base_url

    def list_installed(self):
        """Return [(name, size_bytes), ...] for installed models, largest first."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"Cannot reach Ollama at {self.base_url}: {e}")
            print("Is Ollama running? Set OLLAMA_URL or pass --url.")
            sys.exit(1)
        models = resp.json().get("models", [])
        models.sort(key=lambda m: m.get("size", 0), reverse=True)
        return [(m["name"], m.get("size", 0)) for m in models]

    def chat(self, model, messages, *, schema=None, temperature=0.7,
             num_predict=2048, think=True, think_effort=None, deadline=None,
             on_think=None):
        """One structured-output chat call, streamed.

        Returns (content_str, thinking_char_count, timed_out).

        - `schema` is a JSON schema sent as Ollama's `format` (structured output).
        - `think_effort` (low/medium/high) is sent as the `think` value; models
          that reject a string level fall back to default thinking.
        - `deadline` is an absolute time.time(); generation is cut off once it
          passes (the caller treats that as losing on time).
        - `on_think(token)` is called for each streamed thinking token.
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature, "num_predict": num_predict},
        }
        if schema is not None:
            payload["format"] = schema
        if not think:
            payload["think"] = False
        elif think_effort:
            payload["think"] = think_effort

        resp = requests.post(f"{self.base_url}/api/chat", json=payload,
                             stream=True, timeout=600)
        if resp.status_code != 200 and think_effort:
            # This model likely rejects a string effort level — retry with default thinking.
            resp.close()
            payload.pop("think", None)
            resp = requests.post(f"{self.base_url}/api/chat", json=payload,
                                 stream=True, timeout=600)
        resp.raise_for_status()

        content, think_chars, timed_out, done_reason = [], 0, False, None
        n_chunks = 0
        for line in resp.iter_lines():
            if deadline is not None and time.time() > deadline:
                timed_out = True
                resp.close()
                break
            if not line:
                continue
            chunk = json.loads(line)
            msg = chunk.get("message", {})
            th = msg.get("thinking")
            if th:
                think_chars += len(th)
                n_chunks += 1
                if on_think:
                    on_think(th)
            tok = msg.get("content")
            if tok:
                content.append(tok)
                n_chunks += 1
            if chunk.get("done", False):
                # "length" means num_predict cut generation off mid-reply —
                # the caller uses this to coach "be briefer" instead of the
                # misleading generic bad-JSON feedback. Ollama doesn't always
                # report it (grammar-constrained runs can die at the cap with
                # "stop"), so also infer it from the chunk count: one streamed
                # chunk is one token.
                done_reason = chunk.get("done_reason")
                if done_reason != "length" and n_chunks >= num_predict:
                    done_reason = "length"
                break
        return "".join(content), think_chars, timed_out, done_reason
