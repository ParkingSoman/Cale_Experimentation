"""Pictures in chat requests. Against a running Cale (scripts/cale_dev_server.py HOME).

usage: python cale/tests/images.py [--base http://127.0.0.1:47899]
An engine that cannot read pictures must say so with a clear 400. An engine that can must read them.
"""
import argparse
import base64
import io
import json
import sys
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:47899")
a = ap.parse_args()
FAIL = []


def call(path, body=None, key=None):
    h = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    req = urllib.request.Request(a.base + path, data=None if body is None else json.dumps(body).encode(), headers=h)
    try:
        r = urllib.request.urlopen(req, timeout=900)
        return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{str(detail)[:1500]}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAIL.append(name)


def err(r):
    return r[0], r[1].get("error", {}).get("code")


def picture():
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (512, 512), "white")
    ImageDraw.Draw(im).ellipse((100, 60, 400, 360), fill=(220, 30, 30))
    b = io.BytesIO()
    im.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


key = call("/cale/connect", {"app": "Images Test"})[1]["api_key"]
caps = call("/v1/models", key=key)[1]["cale"].get("capabilities", {})
check("/v1/models says whether pictures work", isinstance(caps.get("images"), bool), caps)


def chat(content, role="user", **kw):
    return call("/v1/chat/completions", {"model": "gemma-3n:fast", "max_tokens": 40, "messages": [{"role": role, "content": content}], **kw}, key)


def img(url):
    return {"type": "image_url", "image_url": {"url": url}}


text = {"type": "text", "text": "What colour is the shape? Answer with one word."}
check("audio input -> 400 unsupported_content", err(chat([text, {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}])) == (400, "unsupported_content"))
if not caps.get("images"):
    check("no picture support: a picture gets 400 unsupported_content", err(chat([text, img("data:image/png;base64,AAAA")])) == (400, "unsupported_content"))
else:
    url = picture()
    r = chat([text, img(url)])
    check("a picture is read", r[0] == 200 and "red" in r[1]["choices"][0]["message"]["content"].lower(), r[1])
    check("...and the prompt counts the picture's tokens", r[1]["usage"]["prompt_tokens"] > 250, r[1].get("usage"))
    r2 = chat([text, img(url)])
    check("a second call with a picture works the same", r2[0] == 200 and "red" in r2[1]["choices"][0]["message"]["content"].lower(), r2[1])
    r = chat("Say hello in three words.")
    check("a text call after a picture call still works", r[0] == 200 and r[1]["choices"][0]["message"]["content"], r)
    r = call("/v1/chat/completions", {"model": "gemma-3n:fast", "max_tokens": 40, "stream": False, "messages": [
        {"role": "user", "content": [text, img(url)]}, {"role": "assistant", "content": "red"}, {"role": "user", "content": "Describe that picture again in one short sentence."}]}, key)
    check("a picture earlier in the chat is kept", r[0] == 200 and r[1]["usage"]["prompt_tokens"] > 250 and "red" in r[1]["choices"][0]["message"]["content"].lower(), r[1])
    check("a web address is refused (no fetching)", err(chat([text, img("https://example.com/a.png")])) == (400, "invalid_image"))
    check("bad base64 -> 400 invalid_image", err(chat([text, img("data:image/png;base64,@@@@")])) == (400, "invalid_image"))
    check("base64 that is not a picture -> 400 invalid_image", err(chat([text, img("data:image/png;base64," + base64.b64encode(b"hello").decode())])) == (400, "invalid_image"))
    check("9 pictures -> 400 invalid_image", err(chat([text] + [img(url)] * 9)) == (400, "invalid_image"))
    check("a picture in a system message -> 400", err(chat([img(url)], role="system"))[0] == 400)
    st = call("/cale/status")[1]
    check("engine healthy afterwards", st.get("last_error") is None, st.get("last_error"))

print("\nall passed" if not FAIL else f"\nFAILED: {FAIL}")
sys.exit(1 if FAIL else 0)
