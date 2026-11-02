"""Synthetic curve stimuli (Curve Detectors style): an arc of a circle, black on white, rotated.
Orientation = direction from the arc's middle point to the circle centre (degrees)."""
import numpy as np
from PIL import Image, ImageDraw

def arc(theta, radius=40, span=110, width=4, size=224, ss=4, cx=None, cy=None, bg=255, fg=0):
    S = size * ss; img = Image.new("L", (S, S), bg); d = ImageDraw.Draw(img)
    cx = size / 2 if cx is None else cx; cy = size / 2 if cy is None else cy
    # arc middle point at (cx, cy); circle centre at distance radius in direction theta
    t = np.deg2rad(theta); ox, oy = cx + radius * np.cos(t), cy - radius * np.sin(t)
    a0 = np.deg2rad(theta + 180 - span / 2); a1 = np.deg2rad(theta + 180 + span / 2)
    ang = np.linspace(a0, a1, 200)
    pts = [((ox + radius * np.cos(a)) * ss, (oy - radius * np.sin(a)) * ss) for a in ang]
    d.line(pts, fill=fg, width=width * ss, joint="curve")
    img = img.resize((size, size), Image.LANCZOS)
    return np.repeat(np.array(img)[:, :, None], 3, 2)

def line(theta, length=80, width=4, size=224, ss=4):
    S = size * ss; img = Image.new("L", (S, S), 255); d = ImageDraw.Draw(img)
    t = np.deg2rad(theta); c = size / 2; dx, dy = length / 2 * np.cos(t), -length / 2 * np.sin(t)
    d.line([((c - dx) * ss, (c - dy) * ss), ((c + dx) * ss, (c + dy) * ss)], fill=0, width=width * ss)
    return np.repeat(np.array(img.resize((size, size), Image.LANCZOS))[:, :, None], 3, 2)

def circle(radius=30, width=4, size=224, ss=4):
    S = size * ss; img = Image.new("L", (S, S), 255); d = ImageDraw.Draw(img); c = size / 2
    d.ellipse([(c - radius) * ss, (c - radius) * ss, (c + radius) * ss, (c + radius) * ss], outline=0, width=width * ss)
    return np.repeat(np.array(img.resize((size, size), Image.LANCZOS))[:, :, None], 3, 2)
