"""Pictures in a chat request. Pure Python (Pillow checks the picture, and only when a picture is sent).

Cale accepts a picture only as a data URL: {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}.
Cale never fetches a web address for you. A request must not make this computer call another server.

| Limit | Value |
|---|---|
| pictures in one request | 8 |
| bytes of one picture (after base64) | 10 MB |
| pixels of one picture | 40 million |

Sound is accepted as {"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}: a WAV file (16-bit PCM,
8-bit or 32-bit float), one clip per request, 30 seconds at most, 10 MB at most.
"""
import base64
import binascii
import io

MAX_IMAGES = 8
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 40_000_000
MAX_AUDIO_SECONDS = 30
MAX_CLIPS = 1
AUDIO_RATE = 16000                 # what the model reads


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


def audio_from_part(part):
    """One {"type": "input_audio", ...} content part -> the WAV bytes (checked)."""
    ia = part.get("input_audio")
    if not isinstance(ia, dict) or not isinstance(ia.get("data"), str):
        raise MediaError('Sound must be {"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}.')
    if str(ia.get("format", "wav")).lower() != "wav":
        raise MediaError("Cale reads WAV sound only. Convert other formats first.")
    if len(ia["data"]) > MAX_BYTES * 4 // 3 + 8:
        raise MediaError(f"A sound clip may be at most {MAX_BYTES // 1024 // 1024} MB.")
    try:
        raw = base64.b64decode(ia["data"], validate=True)
    except (binascii.Error, ValueError):
        raise MediaError("The sound is not valid base64.")
    seconds = _wav_seconds(raw)
    if seconds > MAX_AUDIO_SECONDS:
        raise MediaError(f"A sound clip may be at most {MAX_AUDIO_SECONDS} seconds. This one is {seconds:.0f} seconds.")
    return raw


def _wav_seconds(raw):
    import wave
    try:
        with wave.open(io.BytesIO(raw)) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        raise MediaError("Cale cannot read this sound. Send a WAV file (PCM).")


def load_audio(raw):
    """WAV bytes -> mono float32 samples at 16 kHz (numpy)."""
    import wave
    import numpy as np
    with wave.open(io.BytesIO(raw)) as w:
        n, ch, width, rate = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
        data = w.readframes(n)
    if width == 2:
        a = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        a = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        a = np.frombuffer(data, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise MediaError("Cale reads 8, 16 and 32 bit WAV sound.")
    if ch > 1:
        a = a[: len(a) // ch * ch].reshape(-1, ch).mean(axis=1)
    if rate != AUDIO_RATE and len(a) > 1:
        t = np.linspace(0, len(a) - 1, int(len(a) * AUDIO_RATE / rate))
        a = np.interp(t, np.arange(len(a)), a).astype(np.float32)
    return a
