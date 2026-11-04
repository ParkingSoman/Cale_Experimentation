"""Drawn images for the dial test. Fixed before any card exists, so a card cannot be fitted to them. Stroke, width, angle, frequency, colour and tone. No text."""
import math

from PIL import Image, ImageDraw

SIZE = 224


def _canvas(ground=(128, 128, 128)):
    im = Image.new("RGB", (SIZE, SIZE), ground)
    return im, ImageDraw.Draw(im)


def grating(angle, period, width, ink=(20, 20, 20), ground=(235, 235, 235)):
    im, d = _canvas(ground)
    a = math.radians(angle)
    ux, uy = math.cos(a), math.sin(a)
    for k in range(-40, 41):
        cx, cy = SIZE / 2 + k * period * -uy, SIZE / 2 + k * period * ux
        d.line((cx - 400 * ux, cy - 400 * uy, cx + 400 * ux, cy + 400 * uy), fill=ink, width=width)
    return im


def edge(angle):
    im, d = _canvas((235, 235, 235))
    a = math.radians(angle)
    ux, uy = math.cos(a), math.sin(a)
    nx, ny = -uy, ux
    c = SIZE / 2
    d.polygon([(c - 400 * ux, c - 400 * uy), (c + 400 * ux, c + 400 * uy), (c + 400 * ux + 400 * nx, c + 400 * uy + 400 * ny), (c - 400 * ux + 400 * nx, c - 400 * uy + 400 * ny)], fill=(20, 20, 20))
    return im


def dots(spacing, radius, ink=(20, 20, 20), ground=(235, 235, 235)):
    im, d = _canvas(ground)
    for x in range(0, SIZE + spacing, spacing):
        for y in range(0, SIZE + spacing, spacing):
            d.ellipse((x - radius, y - radius, x + radius, y + radius), fill=ink)
    return im


def rings(period, width):
    im, d = _canvas((235, 235, 235))
    c = SIZE // 2
    for r in range(period, SIZE, period):
        d.ellipse((c - r, c - r, c + r, c + r), outline=(20, 20, 20), width=width)
    return im


def checks(period):
    im, d = _canvas((235, 235, 235))
    for x in range(0, SIZE, period):
        for y in range(0, SIZE, period):
            if (x // period + y // period) % 2:
                d.rectangle((x, y, x + period, y + period), fill=(20, 20, 20))
    return im


def curves(radius, width):
    im, d = _canvas((235, 235, 235))
    for x in range(-radius, SIZE + radius, 2 * radius + 8):
        for y in range(-radius, SIZE + radius, 2 * radius + 8):
            d.arc((x, y, x + 2 * radius, y + 2 * radius), 200, 340, fill=(20, 20, 20), width=width)
    return im


def patch(colour):
    return Image.new("RGB", (SIZE, SIZE), colour)


def split(left, right):
    im = Image.new("RGB", (SIZE, SIZE), left)
    ImageDraw.Draw(im).rectangle((SIZE // 2, 0, SIZE, SIZE), fill=right)
    return im


ITEMS = [
    ("thin lines, level", grating(0, 12, 2)), ("thin lines, upright", grating(90, 12, 2)), ("thin lines, slanted right", grating(45, 12, 2)), ("thin lines, slanted left", grating(135, 12, 2)),
    ("wide bars, level", grating(0, 40, 18)), ("wide bars, upright", grating(90, 40, 18)), ("wide bars, slanted", grating(45, 40, 18)), ("fine lines, close", grating(30, 5, 1)),
    ("one edge, level", edge(0)), ("one edge, upright", edge(90)), ("one edge, slanted", edge(45)), ("dots, small and close", dots(14, 3)),
    ("dots, large and far apart", dots(44, 12)), ("rings", rings(22, 3)), ("checks, small", checks(10)), ("checks, large", checks(36)),
    ("arcs", curves(24, 3)), ("plain red", patch((200, 40, 40))), ("plain green", patch((50, 170, 70))), ("plain blue", patch((50, 80, 200))),
    ("plain orange", patch((230, 140, 40))), ("red against green", split((200, 40, 40), (50, 170, 70))), ("blue against orange", split((50, 80, 200), (230, 140, 40))), ("dark against light", split((30, 30, 30), (225, 225, 225))),
]
