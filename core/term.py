"""Shared ANSI terminal colors (the white-vs-cyber-green identity from ai_chess)."""

CYBER = "\033[38;5;46m"       # neon "cyber green" — player 2 / Black
HILITE_BG = "\033[48;5;238m"  # grey background marking the last action
WHITE = "\033[97m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# Text-presentation selector: stops terminals from rendering glyphs as
# fixed-color emoji, so ANSI colors actually apply.
VS_TEXT = "︎"


def utf8_stdout():
    """Unicode glyphs need UTF-8 stdout (Windows consoles default to cp1252)."""
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
