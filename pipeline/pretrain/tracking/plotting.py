"""
Plot conventions shared by the [T] scripts, so that a model keeps its colour and marker from
one figure to the next: the i-th model given to a script is drawn with style(i), the classical
baseline with BASE_STYLE, and samples are ordered by collision energy (sample_key).
"""
import re

COLORS = ("tab:blue", "tab:green", "tab:purple", "tab:red", "tab:brown", "tab:pink")
MARKERS = ("o", "d", "^", "v", "P", "X")
BASE_STYLE = ("tab:orange", "s--")              # baseline tracking: colour, marker + dashed line


def style(i):
    """(colour, marker + solid line) of the i-th model; cycles after six."""
    return COLORS[i % len(COLORS)], MARKERS[i % len(MARKERS)] + "-"


def sample_key(name):
    """Order samples by collision energy (the number before 'gev'), others after, by name."""
    m = re.search(r"(\d+)gev", name)
    return (int(m.group(1)) if m else 10**9, name)
