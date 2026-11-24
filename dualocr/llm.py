"""Optional cloud tie-breaker for the hardest words (off by default).

Only small crops of the doubtful words leave the computer: the line around
the word, with the word boxed in red, plus the candidate spellings. The key is
stored in the system's password store (macOS Keychain, Windows Credential
Manager, or the Secret Service keyring on Linux), never in a file.
"""
from __future__ import annotations

import base64
import io
import json
import re
from typing import List, Optional

import httpx
from PIL import Image, ImageDraw

from .config import KEYCHAIN_SERVICE

CLAUDE_DEFAULT = "claude-opus-5-5"
PROMPT = (
    "This image is a cropped line from a scanned document (Hebrew and/or English). "
    "One word is marked with a red box. Transcribe ONLY the text inside the red box exactly as printed, "
    "character for character, including any punctuation attached to it (quotes, gershayim, periods, "
    "commas, parentheses, hyphens). Do NOT correct spelling or grammar - if the print has a typo, keep it. "
    "Possible readings from OCR engines: {cands}. They may all be wrong. "
    'Reply with JSON only: {{"text": "<exact text>"}}'
)


def get_key(provider: str) -> Optional[str]:
    try:
        import keyring

        return keyring.get_password(KEYCHAIN_SERVICE, provider)
    except Exception:
        return None


def set_key(provider: str, key: str) -> None:
    import keyring

    if key:
        keyring.set_password(KEYCHAIN_SERVICE, provider, key)
    else:
        try:
            keyring.delete_password(KEYCHAIN_SERVICE, provider)
        except Exception:
            pass


def marked_crop(page: Image.Image, line_box, word_box) -> Image.Image:
    x0, y0, x1, y1 = line_box
    h = y1 - y0
    pad = int(h * 0.6) + 8
    box = (max(0, int(x0 - pad)), max(0, int(y0 - pad)), min(page.width, int(x1 + pad)),
           min(page.height, int(y1 + pad)))
    crop = page.crop(box).convert("RGB")
    d = ImageDraw.Draw(crop)
    wx0, wy0, wx1, wy1 = word_box
    d.rectangle((wx0 - box[0] - 4, wy0 - box[1] - 4, wx1 - box[0] + 4, wy1 - box[1] + 4),
                outline=(230, 0, 0), width=max(2, int(h / 18)))
    if crop.height < 120:
        f = 120 / crop.height
        crop = crop.resize((int(crop.width * f), 120), Image.LANCZOS)
    return crop


def _b64(img: Image.Image) -> str:
    b = io.BytesIO()
    img.save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode()


def _parse(text: str) -> Optional[str]:
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return str(json.loads(m.group(0)).get("text", "")).strip()
        except Exception:
            pass
    return text.strip().strip('"') or None


def ask_claude(key: str, model: str, img: Image.Image, cands: List[str], timeout=60) -> Optional[str]:
    r = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        json={"model": model or CLAUDE_DEFAULT, "max_tokens": 200,
              "messages": [{"role": "user", "content": [
                  {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _b64(img)}},
                  {"type": "text", "text": PROMPT.format(cands=json.dumps(cands, ensure_ascii=False))}]}]},
        timeout=timeout)
    r.raise_for_status()
    data = r.json()
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    return _parse(text)


def gemini_pick_model(key: str) -> str:
    """Newest Gemini 'pro' model that can generate content."""
    try:
        r = httpx.get("https://generativelanguage.googleapis.com/v1beta/models", params={"key": key}, timeout=30)
        r.raise_for_status()
        names = [m["name"].split("/")[-1] for m in r.json().get("models", [])
                 if "generateContent" in m.get("supportedGenerationMethods", [])]
        def ver(n):
            m = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
            return float(m.group(1)) if m else 0.0
        pro = [n for n in names if "pro" in n and "vision" not in n and "exp" not in n]
        pool = pro or [n for n in names if n.startswith("gemini")]
        if pool:
            return sorted(pool, key=lambda n: (ver(n), "preview" not in n))[-1]
    except Exception:
        pass
    return "gemini-2.5-pro"


def ask_gemini(key: str, model: str, img: Image.Image, cands: List[str], timeout=60) -> Optional[str]:
    model = model or gemini_pick_model(key)
    r = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": key},
        json={"contents": [{"parts": [
            {"inline_data": {"mime_type": "image/png", "data": _b64(img)}},
            {"text": PROMPT.format(cands=json.dumps(cands, ensure_ascii=False))}]}],
            "generationConfig": {"temperature": 0}},
        timeout=timeout)
    r.raise_for_status()
    parts = r.json()["candidates"][0]["content"]["parts"]
    return _parse("".join(p.get("text", "") for p in parts))


def ask(provider: str, key: str, model: str, img: Image.Image, cands: List[str]) -> Optional[str]:
    if provider == "gemini":
        return ask_gemini(key, model, img, cands)
    return ask_claude(key, model, img, cands)


def test_key(provider: str, key: str, model: str = "") -> tuple:
    """(ok, message) - makes one tiny request."""
    img = Image.new("RGB", (220, 60), "white")
    ImageDraw.Draw(img).text((20, 20), "test", fill="black")
    try:
        out = ask(provider, key, model, img, ["test"])
        return True, f"Key works (model answered: {out!r})"
    except httpx.HTTPStatusError as e:
        return False, f"The service refused the request ({e.response.status_code}): {e.response.text[:200]}"
    except Exception as e:
        return False, f"Could not reach the service: {e}"
