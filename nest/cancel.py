"""Stopping a request whose client has gone away. Shared by both engines and the server.

A `Cancel` is made for each request. The server sets it when the client closes its connection. The engines check
`is_set()` between pieces of work (every token, every piece of the prompt) and raise `Cancelled`.
A long step that cannot check (llama.cpp reading a prompt) is stopped through a hook: `on_set(fn)` runs `fn` when it is set.
"""
import threading


class Cancelled(Exception):
    """The client left. The work stops. Nothing is wrong with the engine."""


class Cancel:
    def __init__(self):
        self._event = threading.Event()
        self._hooks = []

    def set(self):
        self._event.set()
        for fn in list(self._hooks):
            try:
                fn()
            except Exception:
                pass

    def is_set(self):
        return self._event.is_set()

    def on_set(self, fn):
        self._hooks.append(fn)
        if self._event.is_set():
            fn()

    def check(self):
        if self._event.is_set():
            raise Cancelled()
