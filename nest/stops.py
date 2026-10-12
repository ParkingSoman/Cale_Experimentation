"""Stop sequences, shared by both engines (MLX and llama.cpp), so `stop` means the same thing everywhere.

    f = StopFilter(["\\n\\n", "END"])
    for piece in pieces:
        text, hit = f.feed(piece)      # text is safe to send now. hit is True when a stop sequence was found.
        send(text)
        if hit:
            break
    send(f.flush())                    # whatever was held back (nothing after a hit)

The filter holds back the last few characters, because a stop sequence can arrive in two pieces.
The stop sequence itself is never sent.
"""


class StopFilter:
    def __init__(self, stops):
        self.stops = [s for s in (stops or []) if s]
        self.hold = max((len(s) for s in self.stops), default=1) - 1
        self.buf = ""
        self.hit = False

    def feed(self, piece):
        """Returns (text to send now, True if a stop sequence was found)."""
        if not self.stops:
            return piece, False
        self.buf += piece
        cut = min((i for i in (self.buf.find(s) for s in self.stops) if i >= 0), default=-1)
        if cut >= 0:
            out, self.buf, self.hit = self.buf[:cut], "", True
            return out, True
        keep = len(self.buf) - self.hold
        if keep <= 0:
            return "", False
        out, self.buf = self.buf[:keep], self.buf[keep:]
        return out, False

    def flush(self):
        out, self.buf = ("" if self.hit else self.buf), ""
        return out
