"""Scrabble on the standard 15x15 board — a vocabulary + spatial + scoring test.

Rules in force (in-house, ~no external deps): the standard premium-square layout,
the standard 98-tile bag MINUS the two blanks (no wildcards), 7-tile racks, the
+50 bonus for playing all seven ("bingo"), and exchange / pass. Words are checked
against the public-domain ENABLE list (data/scrabble_words.txt). There is no
challenge phase — an invalid word is simply an illegal move, so the engine
re-prompts (an illegal move only forfeits after the retry budget is spent).

A move is JSON: {"action": "play"|"exchange"|"pass", "square": "H8",
"dir": "across"|"down", "word": "QUILT", "tiles": "AEI", "comment": "..."}.
For a play, `square` is where the word's FIRST letter sits and `word` is the
whole word read along `dir`, INCLUDING any tiles already on the board it runs
through. Parsing is forgiving about case, "H8"/"8H" order, and stray punctuation.

The game ends when a player empties their rack with the bag empty, or after six
successive scoreless turns; unplayed rack tiles are then deducted (and handed to
the player who went out), Scrabble-style."""

import os
import re
from collections import Counter
from datetime import date

from core.game import Game
from core.term import BOLD, CYBER, DIM, HILITE_BG, RESET, WHITE, YELLOW

SIZE = 15
COLS = "ABCDEFGHIJKLMNO"

LETTER_VALUES = {
    "A": 1, "B": 3, "C": 3, "D": 2, "E": 1, "F": 4, "G": 2, "H": 4, "I": 1,
    "J": 8, "K": 5, "L": 1, "M": 3, "N": 1, "O": 1, "P": 3, "Q": 10, "R": 1,
    "S": 1, "T": 1, "U": 1, "V": 4, "W": 4, "X": 8, "Y": 4, "Z": 10,
}

# Standard tile distribution minus the two blanks → 98 tiles.
TILE_DISTRIBUTION = {
    "A": 9, "B": 2, "C": 2, "D": 4, "E": 12, "F": 2, "G": 3, "H": 2, "I": 9,
    "J": 1, "K": 1, "L": 4, "M": 2, "N": 6, "O": 8, "P": 2, "Q": 1, "R": 6,
    "S": 4, "T": 6, "U": 4, "V": 2, "W": 2, "X": 1, "Y": 2, "Z": 1,
}

RACK_SIZE = 7

# Canonical premium-square layout. T=triple word, d=double word (center is the
# star), t=triple letter, l=double letter, .=plain.
_PREMIUM_ROWS = [
    "T..l...T...l..T",
    ".d...t...t...d.",
    "..d...l.l...d..",
    "l..d...l...d..l",
    "....d.....d....",
    ".t...t...t...t.",
    "..l...l.l...l..",
    "T..l...d...l..T",
    "..l...l.l...l..",
    ".t...t...t...t.",
    "....d.....d....",
    "l..d...l...d..l",
    "..d...l.l...d..",
    ".d...t...t...d.",
    "T..l...T...l..T",
]
TW, DW, TL, DL = set(), set(), set(), set()
for _r, _row in enumerate(_PREMIUM_ROWS):
    for _c, _ch in enumerate(_row):
        {"T": TW, "d": DW, "t": TL, "l": DL}.get(_ch, set()).add((_r, _c))
CENTER = (7, 7)

MOVE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["play", "exchange", "pass"]},
        "square": {"type": "string"},
        "dir": {"type": "string", "enum": ["across", "down"]},
        "word": {"type": "string"},
        "tiles": {"type": "string"},
        "comment": {"type": "string"},
    },
    "required": ["action", "comment"],
}

SYSTEM_TMPL = (
    "You are an expert Scrabble player playing a real game for the highest "
    "score. The board is the standard 15x15 with premium squares: 2L/3L double "
    "or triple a single letter's value; 2W/3W double or triple the whole word "
    "(premiums apply only to freshly placed tiles, and only on the turn they "
    "are first covered). Tile values are standard; this set has NO blank tiles. "
    "Playing all seven of your tiles in one move scores a +50 bonus.\n"
    "On your turn you may:\n"
    '- PLAY a word: {"action":"play","square":"<start>","dir":"across"|"down",'
    '"word":"<WORD>"}. `square` is where the FIRST letter goes (e.g. H8); '
    "`word` is the complete word read in that direction, INCLUDING letters "
    "already on the board that it runs through. The first move must cross the "
    "center (H8); every later word must connect to tiles already on the board, "
    "and EVERY word it forms (the main word and any crossing words) must be a "
    "valid dictionary word.\n"
    '- EXCHANGE tiles: {"action":"exchange","tiles":"<letters>"} (only when at '
    "least 7 tiles remain in the bag); you forfeit your score for the turn.\n"
    '- PASS: {"action":"pass"}.\n'
    'Always include a short PRIVATE "comment" (your opponent never sees it). '
    "Reply with ONLY the JSON object. An illegal move wastes your turn and, if "
    "you keep failing, forfeits the game — so make sure your word is real and "
    "your tiles are on your rack."
)


def parse_square(s):
    """'H8' or '8H' (any case, stray punctuation ok) -> (row, col) 0-indexed,
    or None. Column letter A-O, row number 1-15."""
    s = (s or "").upper()
    letters = re.findall(r"[A-O]", s)
    nums = re.findall(r"\d+", s)
    if len(letters) != 1 or len(nums) != 1:
        return None
    col = COLS.index(letters[0])
    row = int(nums[0]) - 1
    if 0 <= row < SIZE and 0 <= col < SIZE:
        return (row, col)
    return None


def sq_name(r, c):
    return f"{COLS[c]}{r + 1}"


def parse_dir(s):
    s = (s or "").strip().lower()
    if s in ("across", "a", "horizontal", "h", "right", "r", "→"):
        return "across"
    if s in ("down", "d", "vertical", "v", "↓"):
        return "down"
    return None


class ScrabbleGame(Game):
    name = "scrabble"
    roles = ("p1", "p2")
    max_rounds_default = 400

    def __init__(self):
        import random
        self.rng = random.Random()
        self.words_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "scrabble_words.txt")
        self.words = None

    # ── configuration ────────────────────────────────────────────────────
    @classmethod
    def add_args(cls, p):
        p.add_argument("--scrabble-words", default=None, metavar="PATH",
                       help="Word list to validate against (default: "
                            "data/scrabble_words.txt, the public-domain ENABLE list)")

    def configure(self, args):
        if getattr(args, "scrabble_words", None):
            self.words_path = args.scrabble_words
        self._load_words()

    def _load_words(self):
        if self.words is not None:
            return
        try:
            with open(self.words_path, encoding="utf-8") as f:
                self.words = {w.strip().upper() for w in f
                              if len(w.strip()) >= 2 and w.strip().isalpha()}
        except OSError as e:
            raise SystemExit(
                f"Scrabble needs a word list at {self.words_path}.\n"
                f"  ({e})\n"
                "  Download the public-domain ENABLE list, e.g.:\n"
                "    curl -fsSL https://raw.githubusercontent.com/dolph/"
                "dictionary/master/enable1.txt \\\n"
                "      | grep -E '^[A-Za-z]{2,15}$' | tr a-z A-Z | sort -u \\\n"
                f"      > {self.words_path}")

    # ── state ────────────────────────────────────────────────────────────
    def initial_state(self):
        self._load_words()
        bag = []
        for letter, n in TILE_DISTRIBUTION.items():
            bag += [letter] * n
        self.rng.shuffle(bag)
        racks = {"p1": [bag.pop() for _ in range(RACK_SIZE)],
                 "p2": [bag.pop() for _ in range(RACK_SIZE)]}
        return {
            "board": [[None] * SIZE for _ in range(SIZE)],
            "racks": racks,
            "bag": bag,
            "scores": {"p1": 0, "p2": 0},
            "turn": "p1",
            "passes": 0,
            "history": [],
            "last_cells": [],
            "over": False,
            "reason": None,
        }

    def current_role(self, state):
        return state["turn"]

    def _other(self, role):
        return "p2" if role == "p1" else "p1"

    def _empty_board(self, state):
        return all(cell is None for row in state["board"] for cell in row)

    # ── prompts / observation ────────────────────────────────────────────
    def system_prompt(self, role):
        return SYSTEM_TMPL

    def _grid_text(self, state):
        """Plain ASCII board for the model: letters where played, premium codes
        (2W/3W/2L/3L) on empty premium squares, '..' elsewhere."""
        board = state["board"]
        head = "    " + " ".join(f"{COLS[c]:>2}" for c in range(SIZE))
        lines = [head]
        for r in range(SIZE):
            cells = []
            for c in range(SIZE):
                v = board[r][c]
                if v is not None:
                    cells.append(f" {v}")
                elif (r, c) in TW:
                    cells.append("3W")
                elif (r, c) in DW:
                    cells.append(" ✸" if (r, c) == CENTER else "2W")
                elif (r, c) in TL:
                    cells.append("3L")
                elif (r, c) in DL:
                    cells.append("2L")
                else:
                    cells.append(" .")
            lines.append(f"{r + 1:>2}  " + " ".join(cells))
        return "\n".join(lines)

    def observation(self, state, role):
        opp = self._other(role)
        rack = "".join(sorted(state["racks"][role]))
        parts = [
            f"You are {role.upper()}. Score: you {state['scores'][role]}, "
            f"opponent {state['scores'][opp]}.",
            f"\nBoard (letters = tiles in play; 3W/2W/3L/2L = empty premium "
            f"squares; ✸ = center; '.' = empty):\n{self._grid_text(state)}",
            f"\nYour rack: {rack}",
            f"\nTiles left in bag: {len(state['bag'])}. "
            f"Opponent holds {len(state['racks'][opp])} tiles.",
        ]
        if self._empty_board(state):
            parts.append("\nThe board is empty — your word must cross the "
                         "center square H8.")
        if state["passes"]:
            parts.append(f"\n{state['passes']} scoreless turn(s) in a row "
                         "(six in a row ends the game).")
        parts.append("\nMake your move now as JSON.")
        return "\n".join(parts)

    def action_schema(self, state, role):
        return MOVE_SCHEMA

    def action_summary(self, action):
        a = str(action.get("action", "?")).lower()
        if a == "play":
            return f"play {action.get('word', '?')}@{action.get('square', '?')}"
        if a == "exchange":
            return f"exchange {action.get('tiles', '?')}"
        return a

    # ── transitions ──────────────────────────────────────────────────────
    def apply(self, state, role, action):
        act = str(action.get("action", "")).strip().lower()
        if act in ("play", "move", "word"):
            return self._do_play(state, role, action)
        if act in ("exchange", "swap", "trade"):
            return self._do_exchange(state, role, action)
        if act in ("pass", "skip"):
            return self._end_turn(state, role, "passes", scoreless=True)
        return "illegal", ('"action" must be "play", "exchange" or "pass"')

    def _do_play(self, state, role, action):
        board = state["board"]
        rack = state["racks"][role]
        anchor = parse_square(action.get("square", ""))
        if anchor is None:
            return "illegal", ('"square" must be a board square like "H8" '
                               "(column A-O, row 1-15)")
        direction = parse_dir(action.get("dir", ""))
        if direction is None:
            return "illegal", '"dir" must be "across" or "down"'
        word = re.sub(r"[^A-Z]", "", str(action.get("word", "")).upper())
        if len(word) < 2:
            return "illegal", "a play must be a word of at least 2 letters"

        across = direction == "across"
        dr, dc = (0, 1) if across else (1, 0)
        r0, c0 = anchor

        # Lay the word out; classify each square as a through-tile or a new tile.
        placements = []        # (r, c, letter) for newly placed tiles
        main_cells = []        # (r, c, letter, is_new) for the whole word
        through = 0
        for i, ch in enumerate(word):
            r, c = r0 + dr * i, c0 + dc * i
            if not (0 <= r < SIZE and 0 <= c < SIZE):
                return "illegal", (f"the word runs off the board (past "
                                   f"{sq_name(r0 + dr * (i - 1), c0 + dc * (i - 1))})")
            existing = board[r][c]
            if existing is not None:
                if existing != ch:
                    return "illegal", (f"{sq_name(r, c)} already holds "
                                       f"'{existing}', but your word needs "
                                       f"'{ch}' there")
                through += 1
                main_cells.append((r, c, ch, False))
            else:
                placements.append((r, c, ch))
                main_cells.append((r, c, ch, True))
        if not placements:
            return "illegal", "you must place at least one new tile"

        # The word must be maximal — no adjoining tiles before or after it.
        pre = (r0 - dr, c0 - dc)
        post = (r0 + dr * len(word), c0 + dc * len(word))
        for rr, cc in (pre, post):
            if 0 <= rr < SIZE and 0 <= cc < SIZE and board[rr][cc] is not None:
                return "illegal", (f"there is already a tile at {sq_name(rr, cc)} "
                                   "adjoining your word — give the COMPLETE word "
                                   "(and its true starting square)")

        # Rack must cover the new tiles.
        need = Counter(ch for _, _, ch in placements)
        have = Counter(rack)
        for ch, n in need.items():
            if have[ch] < n:
                return "illegal", (f"your rack ({''.join(sorted(rack))}) does not "
                                   f"have {n}×'{ch}'")

        first = self._empty_board(state)
        if first:
            if CENTER not in [(r, c) for r, c, _ in placements] and \
               CENTER not in [(r, c) for r, c, _, _ in main_cells]:
                return "illegal", "the first word must cover the center square H8"
        else:
            touches = through > 0 or any(
                self._touches_existing(board, r, c) for r, c, _ in placements)
            if not touches:
                return "illegal", ("your word must connect to tiles already on "
                                   "the board")

        # Gather every word formed (main + crosses), validate, and score.
        main_word = word
        formed = [(main_word, main_cells)]
        for r, c, ch in placements:
            cross = self._cross_cells(board, r, c, ch, across)
            if len(cross) >= 2:
                formed.append(("".join(x[2] for x in cross), cross))

        bad = [w for w, _ in formed if w not in self.words]
        if bad:
            uniq = ", ".join(f'"{w}"' for w in dict.fromkeys(bad))
            return "illegal", (f"{uniq} is not in the dictionary"
                               if len(bad) == 1 else
                               f"these are not in the dictionary: {uniq}")

        score = sum(self._score_word(cells) for _, cells in formed)
        if len(placements) == RACK_SIZE:
            score += 50  # bingo

        # Commit.
        for r, c, ch in placements:
            board[r][c] = ch
        for ch in (p[2] for p in placements):
            rack.remove(ch)
        self._draw(state, role)
        state["scores"][role] += score
        state["last_cells"] = [(r, c) for r, c, _ in placements]
        bingo = " (incl. +50 BINGO)" if len(placements) == RACK_SIZE else ""
        extra = ""
        if len(formed) > 1:
            extra = " (+" + ", ".join(w for w, _ in formed[1:]) + ")"
        state["history"].append(
            f"{role.upper()} {sq_name(*anchor)} {direction} {main_word} "
            f"+{score}{bingo} ({state['scores']['p1']}–{state['scores']['p2']})")
        display = f"{sq_name(*anchor)} {direction} {main_word} +{score}{bingo}{extra}"
        self._end_turn(state, role, None, scoreless=False)
        return "ok", display

    def _touches_existing(self, board, r, c):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < SIZE and 0 <= cc < SIZE and board[rr][cc] is not None:
                return True
        return False

    def _cross_cells(self, board, r, c, letter, across):
        """The word crossing perpendicular to the main direction through the new
        tile at (r,c). Returns cells (r,c,letter,is_new) using the pre-move board
        plus this single new tile; length <2 means no cross word."""
        dr, dc = (1, 0) if across else (0, 1)   # perpendicular axis
        rr, cc = r, c
        while True:
            pr, pc = rr - dr, cc - dc
            if 0 <= pr < SIZE and 0 <= pc < SIZE and board[pr][pc] is not None:
                rr, cc = pr, pc
            else:
                break
        cells = []
        while 0 <= rr < SIZE and 0 <= cc < SIZE and \
                (board[rr][cc] is not None or (rr, cc) == (r, c)):
            is_new = (rr, cc) == (r, c)
            cells.append((rr, cc, letter if is_new else board[rr][cc], is_new))
            rr += dr
            cc += dc
        return cells

    def _score_word(self, cells):
        total, word_mult = 0, 1
        for r, c, letter, is_new in cells:
            lv = LETTER_VALUES[letter]
            if is_new:
                if (r, c) in TL:
                    lv *= 3
                elif (r, c) in DL:
                    lv *= 2
                if (r, c) in TW:
                    word_mult *= 3
                elif (r, c) in DW:
                    word_mult *= 2
            total += lv
        return total * word_mult

    def _do_exchange(self, state, role, action):
        rack = state["racks"][role]
        if len(state["bag"]) < RACK_SIZE:
            return "illegal", (f"you can only exchange when at least {RACK_SIZE} "
                               f"tiles remain in the bag (only {len(state['bag'])} "
                               "left) — play or pass instead")
        tiles = re.sub(r"[^A-Z]", "", str(action.get("tiles", "")).upper())
        if not tiles:
            return "illegal", '"tiles" must list the rack letters to exchange'
        need, have = Counter(tiles), Counter(rack)
        for ch, n in need.items():
            if have[ch] < n:
                return "illegal", (f"you cannot exchange {n}×'{ch}' — your rack is "
                                   f"{''.join(sorted(rack))}")
        for ch in tiles:
            rack.remove(ch)
        state["bag"] += list(tiles)
        self.rng.shuffle(state["bag"])
        self._draw(state, role)
        state["last_cells"] = []
        state["history"].append(f"{role.upper()} exchanged {len(tiles)} tile(s)")
        self._end_turn(state, role, None, scoreless=True)
        return "ok", f"exchanges {len(tiles)} tile(s)"

    def _draw(self, state, role):
        rack = state["racks"][role]
        while len(rack) < RACK_SIZE and state["bag"]:
            rack.append(state["bag"].pop())

    def _end_turn(self, state, role, display, scoreless):
        if scoreless:
            state["passes"] += 1
            if display:
                state["last_cells"] = []
                state["history"].append(f"{role.upper()} {display}")
        else:
            state["passes"] = 0
        # End conditions.
        if not state["bag"] and not state["racks"][role]:
            state["over"], state["reason"] = True, f"{role.upper()} went out"
        elif state["passes"] >= 6:
            state["over"], state["reason"] = True, "six successive scoreless turns"
        state["turn"] = self._other(role)
        if display and scoreless:
            return "ok", display
        return None

    def is_over(self, state):
        return state["over"]

    # ── presentation / results ───────────────────────────────────────────
    def render(self, state):
        board = state["board"]
        last = set(state["last_cells"])
        rows = ["   " + " ".join(f"{DIM}{COLS[c]:>2}{RESET}" for c in range(SIZE))]
        for r in range(SIZE):
            cells = []
            for c in range(SIZE):
                bg = HILITE_BG if (r, c) in last else ""
                v = board[r][c]
                if v is not None:
                    cells.append(f"{bg}{BOLD}{WHITE}{v:>2}{RESET}")
                elif (r, c) in TW:
                    cells.append(f"{bg}{YELLOW}3W{RESET}")
                elif (r, c) in DW:
                    cells.append(f"{bg}{YELLOW}{' ✸' if (r, c) == CENTER else '2W'}{RESET}")
                elif (r, c) in TL:
                    cells.append(f"{bg}{CYBER}3L{RESET}")
                elif (r, c) in DL:
                    cells.append(f"{bg}{CYBER}2L{RESET}")
                else:
                    cells.append(f"{bg}{DIM} ·{RESET}")
            rows.append(f"{DIM}{r + 1:>2}{RESET} " + " ".join(cells))
        rows.append(f"   {DIM}p1 {state['scores']['p1']}  ·  "
                    f"p2 {state['scores']['p2']}  ·  bag {len(state['bag'])}{RESET}")
        return "\n".join(rows)

    def _rack_value(self, rack):
        return sum(LETTER_VALUES[t] for t in rack)

    def result(self, state, forfeit=None, capped=False):
        scores = dict(state["scores"])
        racks = state["racks"]
        if forfeit is not None:
            role, kind = forfeit
            winner = self._other(role)
            why = "ran out of time" if kind == "time" else "failed to move"
            summary = f"forfeit ({role.upper()} {why})"
        else:
            # Endgame tile adjustment.
            out = next((r for r in self.roles
                        if not racks[r] and not state["bag"]), None)
            if out is not None:
                opp = self._other(out)
                scores[out] += self._rack_value(racks[opp])
                scores[opp] -= self._rack_value(racks[opp])
            else:
                for r in self.roles:
                    scores[r] -= self._rack_value(racks[r])
            if scores["p1"] > scores["p2"]:
                winner = "p1"
            elif scores["p2"] > scores["p1"]:
                winner = "p2"
            else:
                winner = None
            how = ("adjudicated at the move cap" if capped
                   else state.get("reason") or "game over")
            if winner:
                summary = (f"{winner.upper()} wins {scores[winner]}–"
                           f"{scores[self._other(winner)]} ({how})")
            else:
                summary = f"tie {scores['p1']}–{scores['p2']} ({how})"
        return {
            "winner": winner,
            "summary": summary,
            "scores": {"p1": 1.0 if winner == "p1" else 0.5 if winner is None else 0.0,
                       "p2": 1.0 if winner == "p2" else 0.5 if winner is None else 0.0},
            "final_scores": scores,
            "no_winner_banner": "drawn game",
        }

    def export(self, state, outcome, run_dir):
        labels = outcome.get("labels", {})
        head = (f"SCRABBLE — {outcome.get('summary', '')}",
                f"  p1: {labels.get('p1', 'p1')}",
                f"  p2: {labels.get('p2', 'p2')}",
                "", "Moves:")
        lines = list(head) + [f"  {i}. {h}" for i, h in
                              enumerate(state["history"], 1)]
        lines += ["", "Final board:", self._grid_text(state)]
        path = os.path.join(run_dir, "game.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return [path]
