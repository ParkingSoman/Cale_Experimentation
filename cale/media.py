"""Pictures in a chat request. Pure Python (Pillow checks the picture, and only when a picture is sent).

Cale accepts a picture only as a data URL: {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}.
Cale never fetches a web address for you. A request must not make this computer call another server.

| Limit | Value |
|---|---|
| pictures in one request | 8 |
| bytes of one picture (after base64) | 10 MB |
| pixels of one picture | 40 million |
"""
import base64
import binascii
import io

MAX_IMAGES = 8
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 40_000_000


class MediaError(ValueError):
    pass


def image_from_part(part):
    """One {"type": "image_url", ...} content part -> the picture's bytes (checked)."""
    iu = part.get("image_url")
    url = iu.get("url") if isinstance(iu, dict) else iu
    if not isinstance(url, str) or not url.startswith("data:image/"):
        raise MediaError("A picture must be a data URL (data:image/png;base64,...). Cale does not fetch web addresses.")
    head, sep, body = url.partition(",")
    if not sep or not head.endswith(";base64"):
        raise MediaError("A picture data URL must be base64 encoded.")
    if len(body) > MAX_BYTES * 4 // 3 + 8:
        raise MediaError(f"A picture may be at most {MAX_BYTES // 1024 // 1024} MB.")
    try:
        raw = base64.b64decode(body, validate=True)
    except (binascii.Error, ValueError):
        raise MediaError("The picture is not valid base64.")
    try:
        from PIL import Image
        with Image.open(io.BytesIO(raw)) as im:
            w, h = im.size
            if w * h > MAX_PIXELS:
                raise MediaError(f"A picture may have at most {MAX_PIXELS // 1_000_000} million pixels. This one has {w * h / 1e6:.0f} million.")
            im.verify()
    except MediaError:
        raise
    except Exception:
        raise MediaError("Cale cannot read this picture. Send a PNG, JPEG, WebP or GIF.")
    return raw


def load(raw):
    """Picture bytes -> an RGB Pillow image."""
    from PIL import Image
    with Image.open(io.BytesIO(raw)) as im:
        return im.convert("RGB")
