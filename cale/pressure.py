"""Memory pressure level: 1 normal, 2 warn, 4 critical. The OS-specific reading lives in oslayer."""
from . import oslayer

NAMES = {1: "normal", 2: "warn", 4: "critical"}
_override = None        # set by the debug endpoint only (CALE_DEBUG=1)


def set_override(level):
    global _override
    _override = level


def level():
    if _override is not None:
        return _override
    return oslayer.pressure_level()


def name():
    return NAMES.get(level(), "normal")
