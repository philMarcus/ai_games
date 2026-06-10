"""Go on a 9×9 board — the spatial-reasoning stress test. Rules are in-house
(~150 lines): place a stone, flood-fill captures, suicide illegal, positional
superko (no whole-board repeats). Two consecutive passes end the game, scored
by Tromp–Taylor area counting (stones + territory surrounded by only one
color) plus komi for White. An illegal move (after retries) forfeits.

Moves are GTP coordinates — columns A–J skipping I, rows 1–9 — plus "pass".
Expect LLMs to find this much harder than chess: liberty/capture tracking from
a text board is exactly the failure mode this game measures."""

import os
import re
from datetime import date

from core.game import Game
from core.term import BOLD, CYBER, DIM, HILITE_BG, RESET, WHITE, YELLOW

SIZE = 9
COLS = "ABCDEFGHJ"               # GTP columns: no letter I
STAR_POINTS = {(2, 2), (6, 2), (4, 4), (2, 6), (6, 6)}

MOVE_SCHEMA = {
    "type": "object",
    "properties": {
        "move": {"type": "string"},
        "comment": {"type": "string"},
    },
    "required": ["move", "comment"],
}

SYSTEM_TMPL = (
    "You are a strong Go player playing a real game as {color} on a 9x9 board.\n"
    "Rules in force:\n"
    "- Stones are placed on empty intersections. A group with no liberties is "
    "captured and removed. Liberties are the empty points directly adjacent "
    "HORIZONTALLY or VERTICALLY — diagonals never count. The board edge is a "
    "wall: an edge stone has at most 3 liberties, a corner stone at most 2.\n"
    "- Suicide (a move that leaves your own group with no liberties without "
    "capturing) is illegal.\n"
    "- Superko: a move may not recreate any previous whole-board position "
    "(this includes simple ko — you must play elsewhere before recapturing).\n"
    "- Two consecutive passes end the game. Scoring is area (Tromp-Taylor): "
    "your stones on the board plus empty points surrounded only by your color; "
    "White additionally gets {komi} komi.\n"
    "Coordinates: column letter A-H or J (there is NO column I) plus row number "
    "1-9, e.g. E5, C3, J9 — or \"pass\".\n"
    'Reply with ONLY a JSON object {{"move": "<coordinate or pass>", '
    '"comment": "<text>"}}. An illegal or impossible move forfeits the game.\n'
    "The \"comment\" is one or two sentences on your plan. It is PRIVATE — your "
    "opponent never sees it.\n"
    "Decide promptly: keep any reasoning brief and commit to a move."
)


def coord_to_xy(s):
    """'E5' -> (4, 4) zero-indexed (x from A, y from row 1). None if invalid."""
    m = re.fullmatch(r"([A-HJa-hj])\s*([1-9])", s.strip())
    if not m:
        return None
    x = COLS.index(m.group(1).upper())
    y = int(m.group(2)) - 1
    return (x, y)


def xy_to_coord(p):
    return f"{COLS[p[0]]}{p[1] + 1}"


class GoBoard:
    """9×9 board mechanics: captures, suicide, positional superko, area scoring."""

    def __init__(self, size=SIZE):
        self.size = size
        self.grid = {}                 # (x,y) -> "b" | "w"
        self.captures = {"b": 0, "w": 0}
        self.history = []              # [(color, (x,y) | None), ...]; None = pass
        self.simple_ko = None          # point hint for the prompt (superko enforces)
        self.seen = {self.position_key()}

    def position_key(self):
        return frozenset(self.grid.items())

    def neighbors(self, p):
        x, y = p
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                yield (nx, ny)

    def group(self, p):
        """Return (stones, liberties) of the group containing p."""
        color = self.grid[p]
        stones, libs, todo = set(), set(), [p]
        while todo:
            q = todo.pop()
            if q in stones:
                continue
            stones.add(q)
            for n in self.neighbors(q):
                c = self.grid.get(n)
                if c is None:
                    libs.add(n)
                elif c == color and n not in stones:
                    todo.append(n)
        return stones, libs

    def play(self, color, point):
        """Apply a move. point=None is a pass. Returns ("ok", captured_count)
        or ("illegal", reason)."""
        if point is None:
            self.history.append((color, None, 0))
            self.simple_ko = None
            return "ok", 0
        if point in self.grid:
            return "illegal", f"{xy_to_coord(point)} is already occupied"

        opp = "w" if color == "b" else "b"
        self.grid[point] = color
        captured = set()
        for n in self.neighbors(point):
            if self.grid.get(n) == opp:
                stones, libs = self.group(n)
                if not libs:
                    captured |= stones
        for q in captured:
            del self.grid[q]

        _, own_libs = self.group(point)
        if not own_libs:
            del self.grid[point]
            for q in captured:                 # (captured can't be non-empty here,
                self.grid[q] = opp             # but restore defensively)
            return "illegal", (f"{xy_to_coord(point)} is suicide — your stone "
                               "would have no liberties")

        key = self.position_key()
        if key in self.seen:
            del self.grid[point]
            for q in captured:
                self.grid[q] = opp
            return "illegal", (f"{xy_to_coord(point)} violates the ko/superko "
                               "rule — it recreates a previous board position")

        self.seen.add(key)
        self.captures[color] += len(captured)
        self.history.append((color, point, len(captured)))
        # Simple-ko hint: single-stone capture by a now-single-stone group whose
        # only liberty is the captured point.
        self.simple_ko = None
        if len(captured) == 1:
            stones, libs = self.group(point)
            cap = next(iter(captured))
            if len(stones) == 1 and libs == {cap}:
                self.simple_ko = cap
        return "ok", len(captured)

    def two_passes(self):
        return (len(self.history) >= 2
                and self.history[-1][1] is None and self.history[-2][1] is None)

    def area_score(self):
        """Tromp-Taylor: stones + empty regions bordering only one color.
        Returns (black_points, white_points) BEFORE komi."""
        score = {"b": 0, "w": 0}
        for c in self.grid.values():
            score[c] += 1
        empties = {(x, y) for x in range(self.size) for y in range(self.size)
                   if (x, y) not in self.grid}
        while empties:
            start = empties.pop()
            region, borders, todo = {start}, set(), [start]
            while todo:
                q = todo.pop()
                for n in self.neighbors(q):
                    c = self.grid.get(n)
                    if c is None:
                        if n not in region:
                            region.add(n)
                            todo.append(n)
                    else:
                        borders.add(c)
            empties -= region
            if borders == {"b"}:
                score["b"] += len(region)
            elif borders == {"w"}:
                score["w"] += len(region)
        return score["b"], score["w"]


# ──────────────────────────────────────────────────────────────────────────
# The game
# ──────────────────────────────────────────────────────────────────────────

class GoGame(Game):
    name = "go"
    roles = ("black", "white")           # Black moves first
    role_colors = {"black": CYBER, "white": WHITE}
    max_rounds_default = 180

    def __init__(self):
        self.komi = 7.0

    @classmethod
    def add_args(cls, p):
        p.add_argument("--komi", type=float, default=7.0,
                       help="White's compensation points, area scoring (default 7.0)")

    def configure(self, args):
        self.komi = args.komi

    # ── state ────────────────────────────────────────────────────────────
    def initial_state(self):
        return {"board": GoBoard(), "comments": [], "last": None}

    def current_role(self, state):
        hist = state["board"].history
        if not hist:
            return "black"
        return "white" if hist[-1][0] == "b" else "black"

    def is_over(self, state):
        return state["board"].two_passes()

    # ── prompts ──────────────────────────────────────────────────────────
    def system_prompt(self, role):
        return SYSTEM_TMPL.format(color=role.capitalize(), komi=self.komi)

    def _history_text(self, board):
        if not board.history:
            return "(none yet — this is the first move)"
        parts = []
        for i, (c, p, caps) in enumerate(board.history, 1):
            mv = xy_to_coord(p) if p else "pass"
            tag = f" (captured {caps})" if caps else ""
            parts.append(f"{i}.{'B' if c == 'b' else 'W'} {mv}{tag}")
        return "  ".join(parts)

    def _board_text(self, board):
        rows = ["   " + " ".join(COLS[:board.size])]
        for y in range(board.size - 1, -1, -1):
            cells = []
            for x in range(board.size):
                c = board.grid.get((x, y))
                cells.append("X" if c == "b" else "O" if c == "w" else ".")
            rows.append(f"{y + 1:>2} " + " ".join(cells) + f" {y + 1}")
        rows.append("   " + " ".join(COLS[:board.size]))
        return "\n".join(rows)

    def observation(self, state, role):
        board = state["board"]
        me = "b" if role == "black" else "w"
        opp = "w" if me == "b" else "b"
        parts = [
            f"You are {role.capitalize()}. It is your move "
            f"(move {len(board.history) + 1}).",
            f"\nMove history:\n{self._history_text(board)}",
            f"\nBoard (X = Black, O = White, . = empty):\n{self._board_text(board)}",
            f"\nCaptures — you: {board.captures[me]}, opponent: {board.captures[opp]}. "
            f"Komi: {self.komi} (White).",
        ]
        if board.history:
            lc, lp, lcaps = board.history[-1]
            if lcaps:
                mv = xy_to_coord(lp)
                parts.append(f"\nNOTE: your opponent's last move ({mv}) CAPTURED "
                             f"{lcaps} of your stone(s) — they have been removed "
                             "from the board.")
        if board.simple_ko is not None:
            parts.append(f"\nKO: playing at {xy_to_coord(board.simple_ko)} is "
                         "currently forbidden (it would retake the ko immediately).")
        if board.history and board.history[-1][1] is None:
            parts.append("\nYour opponent just PASSED. If you also pass, the game "
                         "ends and is scored as it stands.")
        parts.append("\nGive your move now as JSON.")
        return "\n".join(parts)

    def action_schema(self, state, role):
        return MOVE_SCHEMA

    def action_summary(self, action):
        return str(action.get("move", "?"))

    # ── transitions ──────────────────────────────────────────────────────
    def apply(self, state, role, action):
        board = state["board"]
        raw = str(action.get("move", "")).strip()
        color = "b" if role == "black" else "w"
        if raw.lower() in ("pass", "pas", "p"):
            board.play(color, None)
            state["last"] = None
            state["comments"].append(str(action.get("comment", "")).strip())
            return "ok", "pass"
        point = coord_to_xy(raw)
        if point is None:
            return "illegal", (f'"{raw}" is not a valid coordinate — use a column '
                               "letter A-H or J (there is no I) plus a row 1-9, "
                               'e.g. "E5", or "pass"')
        verdict, info = board.play(color, point)
        if verdict == "illegal":
            return "illegal", info
        state["last"] = point
        state["comments"].append(str(action.get("comment", "")).strip())
        cap = f" {YELLOW}(captured {info}){RESET}" if info else ""
        return "ok", f"{xy_to_coord(point)}{cap}"

    # ── presentation / results ───────────────────────────────────────────
    def render(self, state):
        board = state["board"]
        rows = []
        for y in range(board.size - 1, -1, -1):
            cells = []
            for x in range(board.size):
                c = board.grid.get((x, y))
                bg = HILITE_BG if state["last"] == (x, y) else ""
                if c == "b":
                    cells.append(f"{bg}{BOLD}{CYBER}●{RESET}")
                elif c == "w":
                    cells.append(f"{bg}{BOLD}{WHITE}○{RESET}")
                elif (x, y) in STAR_POINTS:
                    cells.append(f"{bg}{DIM}+{RESET}")
                else:
                    cells.append(f"{bg}{DIM}·{RESET}")
            rows.append(f"{DIM}{y + 1:>2}{RESET} " + " ".join(cells))
        rows.append(f"   {DIM}{' '.join(COLS[:board.size])}{RESET}")
        rows.append(f"   {DIM}captures — black: {board.captures['b']}  "
                    f"white: {board.captures['w']}{RESET}")
        return "\n".join(rows)

    def result(self, state, forfeit=None, capped=False):
        board = state["board"]
        sb, sw = board.area_score()
        sw_komi = sw + self.komi
        if forfeit is not None:
            role, kind = forfeit
            winner = "white" if role == "black" else "black"
            why = "ran out of time" if kind == "time" else "illegal move"
            summary = f"forfeit ({role.capitalize()} {why})"
        else:
            if sb > sw_komi:
                winner, margin = "black", sb - sw_komi
            elif sw_komi > sb:
                winner, margin = "white", sw_komi - sb
            else:
                winner, margin = None, 0
            verdict = (f"{'B' if winner == 'black' else 'W'}+{margin:g}"
                       if winner else "jigo (drawn)")
            how = "adjudicated at move cap" if capped else "two passes"
            summary = (f"{verdict} — black {sb} vs white {sw}+{self.komi} komi "
                       f"({how})")
        return {"winner": winner, "summary": summary,
                "scores": {"black": float(sb), "white": float(sw_komi)},
                "moves": len(board.history),
                "no_winner_banner": "jigo — drawn game"}

    def export(self, state, outcome, run_dir):
        """SGF with each model's comment attached to its move."""
        board = state["board"]
        sgf_alpha = "abcdefghi"

        def sgf_coord(p):
            if p is None:
                return ""
            return sgf_alpha[p[0]] + sgf_alpha[board.size - 1 - p[1]]

        def esc(s):
            return s.replace("\\", "\\\\").replace("]", "\\]")

        nodes = []
        for i, (c, p, _caps) in enumerate(board.history):
            node = f";{'B' if c == 'b' else 'W'}[{sgf_coord(p)}]"
            if i < len(state["comments"]) and state["comments"][i]:
                node += f"C[{esc(state['comments'][i])}]"
            nodes.append(node)

        header = (f"(;GM[1]FF[4]CA[UTF-8]SZ[{board.size}]KM[{self.komi}]"
                  f"PB[{esc(outcome['labels']['black'])}]"
                  f"PW[{esc(outcome['labels']['white'])}]"
                  f"DT[{date.today().isoformat()}]"
                  f"RE[{esc(outcome.get('summary', '?'))}]")
        path = os.path.join(run_dir, "game.sgf")
        with open(path, "w", encoding="utf-8") as f:
            f.write(header + "".join(nodes) + ")\n")
        return [path]
