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

# Distinct bright foreground colors for tagging individual speakers in a
# transcript, so each player's name reads in its own color. Cycled by seat.
NAME_PALETTE = (
    "\033[38;5;81m",   # sky blue
    "\033[38;5;213m",  # pink
    "\033[38;5;220m",  # gold
    "\033[38;5;120m",  # light green
    "\033[38;5;208m",  # orange
    "\033[38;5;147m",  # periwinkle
    "\033[38;5;204m",  # rose
    "\033[38;5;51m",   # aqua
    "\033[38;5;229m",  # pale yellow
    "\033[38;5;156m",  # spring green
)

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
