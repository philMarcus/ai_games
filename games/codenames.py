"""Codenames, cooperative role-split: one model is the SPYMASTER (sees the
hidden color map, gives one-word clues), the other is the GUESSER (sees only
the words, guesses ONE at a time and may stop to bank the turn). Together they
try to find all 9 target words within a turn budget without ever touching the
assassin. Tests theory of mind — the cluegiver must model what the guesser
will infer — plus semantic precision and risk management.

Board: 25 words from data/codenames_words.txt — 9 targets, 15 neutral, 1
assassin. Neutral guess ends the turn; the assassin ends the game. Clues are
validated programmatically (one word, not a board word or derivative)."""

import os
import random
import re

from core.game import Game
from core.term import BOLD, CYBER, DIM, GREEN, RED, RESET, YELLOW

TARGETS = 9
BOARD_SIZE = 25

CLUE_SCHEMA = {
    "type": "object",
    "properties": {
        "clue": {"type": "string"},
        "count": {"type": "integer", "minimum": 1, "maximum": 9},
        "comment": {"type": "string"},
    },
    "required": ["clue", "count", "comment"],
}

GUESS_SCHEMA = {
    "type": "object",
    "properties": {
        "guess": {"type": "string"},
        "comment": {"type": "string"},
    },
    "required": ["guess", "comment"],
}

SPYMASTER_SYSTEM = (
    "You are the SPYMASTER in a cooperative word game. Your partner (another AI) "
    "sees the same 25-word board but NOT the colors. Each turn you give a clue: "
    "ONE single word plus a number — the number of board words your clue points "
    "to. Your partner then guesses words one at a time.\n"
    "Rules for the clue: exactly one word, about the MEANINGS of words; it must "
    "not be (or contain, or be contained in) any unrevealed word on the board.\n"
    "Guide your partner to all TARGET words. If they ever pick the ASSASSIN "
    "word, you both lose instantly — steer well clear of clues that could point "
    "anywhere near it. Neutral picks waste a turn, and turns are limited.\n"
    "The \"comment\" field is PRIVATE — your partner never sees it."
)

GUESSER_SYSTEM = (
    "You are the GUESSER in a cooperative word game. Your partner (another AI) "
    "sees which of the 25 board words are targets; you only see the words. Each "
    "turn they give you a clue word and a number (how many board words it points "
    "to). You guess ONE word at a time:\n"
    "- a TARGET: it is revealed, and you may keep guessing (up to the number "
    "plus one bonus guess)\n"
    "- a NEUTRAL word: your turn ends\n"
    "- the ASSASSIN: you both lose INSTANTLY — if a word might be it, don't.\n"
    "After any correct guess you may also reply STOP to bank the turn instead "
    "of risking another guess. Turns are limited, so use the clue's full count "
    "when you're confident.\n"
    "The \"comment\" field is PRIVATE — your partner never sees it."
)


def load_words(path):
    with open(path, encoding="utf-8") as f:
        return [w.strip().upper() for w in f if w.strip()]


class CodenamesGame(Game):
    name = "codenames"
    roles = ("spymaster", "guesser")
    max_rounds_default = 200

    def __init__(self):
        self.turn_limit = 9
        self.rng = random.Random()
        self.words_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "codenames_words.txt")

    @classmethod
    def add_args(cls, p):
        p.add_argument("--turns", type=int, default=9,
                       help="Clue turns the team gets to find all 9 targets (default 9)")

    def configure(self, args):
        self.turn_limit = args.turns

    # ── state ────────────────────────────────────────────────────────────
    def initial_state(self):
        pool = load_words(self.words_path)
        words = self.rng.sample(pool, BOARD_SIZE)
        shuffled = words[:]
        self.rng.shuffle(shuffled)
        kinds = {}
        for i, w in enumerate(shuffled):
            kinds[w] = ("target" if i < TARGETS
                        else "assassin" if i == TARGETS else "neutral")
        return {"words": words, "kinds": kinds, "revealed": {},
                "turn": 1, "clue": None, "count": 0, "guesses_left": 0,
                "guessed_this_turn": 0, "found": 0,
                "log": [],            # per turn: {"clue","count","guesses":[(w,kind)]}
                "over": False, "won": False, "assassin_hit": None}

    def current_role(self, state):
        return "spymaster" if state["clue"] is None else "guesser"

    def is_over(self, state):
        return state["over"] or state["turn"] > self.turn_limit

    # ── prompts ──────────────────────────────────────────────────────────
    def system_prompt(self, role):
        return SPYMASTER_SYSTEM if role == "spymaster" else GUESSER_SYSTEM

    def _board_for(self, state, role):
        if role == "spymaster":
            return self._secret_map(state)
        rows = []
        for w in state["words"]:
            if w in state["revealed"]:
                rows.append(f"  {w} — REVEALED ({state['revealed'][w]})")
            else:
                rows.append(f"  {w}")
        return "\n".join(rows)

    def _secret_map(self, state):
        """Spymaster view: grouped by kind, hidden vs found — far easier to
        clue from than the guesser's board order."""
        def words(kind, revealed):
            return [w for w in state["words"]
                    if state["kinds"][w] == kind
                    and (w in state["revealed"]) == revealed]

        assassin = words("assassin", False) + words("assassin", True)
        lines = ["YOUR SECRET MAP:",
                 f"  TARGETS still hidden ({len(words('target', False))}): "
                 + (", ".join(words("target", False)) or "(none)"),
                 f"  targets already found: "
                 + (", ".join(words("target", True)) or "(none)"),
                 f"  neutral, unrevealed: "
                 + (", ".join(words("neutral", False)) or "(none)"),
                 f"  neutral, hit so far: "
                 + (", ".join(words("neutral", True)) or "(none)"),
                 f"  THE ASSASSIN: {assassin[0]} — your partner must NEVER pick this"]
        return "\n".join(lines)

    def _history(self, state):
        if not state["log"]:
            return "(no clues yet)"
        out = []
        for i, t in enumerate(state["log"], 1):
            gs = ", ".join(f"{w} ({k})" for w, k in t["guesses"]) or "(no guesses)"
            out.append(f"Turn {i}: clue \"{t['clue']}\" {t['count']} → {gs}")
        return "\n".join(out)

    def observation(self, state, role):
        left = TARGETS - state["found"]
        turns_left = self.turn_limit - state["turn"] + 1
        if role == "spymaster":
            return (f"Turn {state['turn']} of {self.turn_limit} "
                    f"({left} target words still hidden).\n"
                    f"\n{self._board_for(state, 'spymaster')}\n"
                    f"\nClue history:\n{self._history(state)}\n"
                    f"\nGive your clue now as JSON: one word and a count "
                    f"(1-{min(9, left)}).")
        guesses_left = state["guesses_left"]
        unrevealed = ", ".join(w for w in state["words"]
                               if w not in state["revealed"])
        this_turn = ""
        if state["log"] and state["log"][-1]["guesses"]:
            got = ", ".join(f"{w} ({k})" for w, k in state["log"][-1]["guesses"])
            this_turn = (f"\nTHIS TURN you already guessed: {got}. Those words are "
                         "now revealed and CANNOT be guessed again — pick a "
                         "DIFFERENT word that also fits the clue, or STOP.\n")
        return (f"Turn {state['turn']} of {self.turn_limit} "
                f"({left} target words still hidden).\n"
                f"\nThe board:\n{self._board_for(state, 'guesser')}\n"
                f"\nClue history:\n{self._history(state)}\n"
                f"\nCurrent clue: \"{state['clue']}\" {state['count']} — you have "
                f"{guesses_left} guess(es) left this turn.\n{this_turn}"
                f"Your guess MUST be one of these unrevealed board words (the clue "
                f"points AT them, it is not itself a guess):\n{unrevealed}\n"
                f"Reply as JSON: \"guess\" is exactly one word from that list"
                + (", or STOP to bank the turn."
                   if state["guessed_this_turn"] else ". You must guess at least "
                   "once before you may STOP."))

    def action_schema(self, state, role):
        return CLUE_SCHEMA if role == "spymaster" else GUESS_SCHEMA

    def action_summary(self, action):
        if "clue" in action:
            return f"{action.get('clue')} {action.get('count')}"
        return str(action.get("guess", "?"))

    # ── transitions ──────────────────────────────────────────────────────
    def _end_turn(self, state):
        state["clue"] = None
        state["count"] = 0
        state["guesses_left"] = 0
        state["guessed_this_turn"] = 0
        state["turn"] += 1

    def apply(self, state, role, action):
        if role == "spymaster":
            clue = str(action.get("clue", "")).strip()
            count = action.get("count")
            if not re.fullmatch(r"[A-Za-z][A-Za-z'-]*", clue):
                return "illegal", "the clue must be exactly one word (letters only)"
            cl = clue.lower()
            for w in state["words"]:
                if w in state["revealed"]:
                    continue
                wl = w.lower()
                if cl == wl or cl in wl or wl in cl:
                    return "illegal", (f'the clue may not be, contain, or be part '
                                       f'of a board word ("{w}" is on the board)')
            left = TARGETS - state["found"]
            if not isinstance(count, int) or not 1 <= count <= min(9, left):
                return "illegal", f"count must be an integer from 1 to {min(9, left)}"
            state["clue"] = clue
            state["count"] = count
            state["guesses_left"] = count + 1
            state["log"].append({"clue": clue, "count": count, "guesses": []})
            return "ok", f'CLUE: "{clue}" {count}'

        # guesser
        g = str(action.get("guess", "")).strip().upper()
        if g == "STOP":
            if not state["guessed_this_turn"]:
                return "illegal", "you must guess at least once before stopping"
            self._end_turn(state)
            return "ok", f"STOP {DIM}(turn banked){RESET}"
        if g not in state["words"]:
            return "illegal", f'"{g}" is not a word on the board'
        if g in state["revealed"]:
            return "illegal", f'"{g}" is already revealed'

        kind = state["kinds"][g]
        state["revealed"][g] = kind
        state["log"][-1]["guesses"].append((g, kind))
        state["guessed_this_turn"] += 1
        state["guesses_left"] -= 1

        if kind == "assassin":
            state["over"] = True
            state["assassin_hit"] = g
            return "ok", f'"{g}" — {RED}{BOLD}THE ASSASSIN. Game over.{RESET}'
        if kind == "target":
            state["found"] += 1
            if state["found"] == TARGETS:
                state["over"] = True
                state["won"] = True
                return "ok", f'"{g}" — {GREEN}target! That was the last one.{RESET}'
            if state["guesses_left"] == 0:
                self._end_turn(state)
                return "ok", f'"{g}" — {GREEN}target!{RESET} (no guesses left, turn ends)'
            return "ok", f'"{g}" — {GREEN}target!{RESET}'
        self._end_turn(state)
        return "ok", f'"{g}" — {YELLOW}neutral, turn ends{RESET}'

    # ── presentation / results ───────────────────────────────────────────
    def render(self, state):
        cells = []
        for w in state["words"]:
            if w in state["revealed"]:
                kind = state["revealed"][w]
                color = (GREEN if kind == "target"
                         else RED if kind == "assassin" else DIM)
                cells.append(f"{color}{w:<12}{RESET}")
            else:
                cells.append(f"{BOLD}{w:<12}{RESET}")
        rows = ["  " + "".join(cells[i:i + 5]) for i in range(0, BOARD_SIZE, 5)]
        status = (f"  {DIM}targets found: {state['found']}/{TARGETS}   "
                  f"turn {min(state['turn'], self.turn_limit)}/{self.turn_limit}{RESET}")
        return "\n".join(rows + [status])

    def result(self, state, forfeit=None, capped=False):
        found, turns = state["found"], len(state["log"])
        if forfeit is not None:
            role, kind = forfeit
            why = "ran out of time" if kind == "time" else "illegal action"
            summary = f"forfeit ({role} {why}) — {found}/{TARGETS} found"
            banner = "team forfeits"
        elif state["assassin_hit"]:
            summary = (f"hit the assassin (\"{state['assassin_hit']}\") on turn "
                       f"{turns} — {found}/{TARGETS} found")
            banner = "ASSASSIN — team loses"
        elif state["won"]:
            summary = f"all {TARGETS} targets found in {turns} turns"
            banner = f"TEAM WINS in {turns} turns"
        else:
            summary = f"out of turns — {found}/{TARGETS} found"
            banner = "team loses (out of turns)"
        return {"winner": None, "summary": summary, "no_winner_banner": banner,
                "scores": {r: float(found) for r in self.roles},
                "extra": {"found": found, "cleared": state["won"],
                          "assassin": bool(state["assassin_hit"]),
                          "turns_used": turns}}

    def export(self, state, outcome, run_dir):
        lines = [f"Codenames: {outcome['labels']['spymaster']} (spymaster) + "
                 f"{outcome['labels']['guesser']} (guesser)",
                 f"result: {outcome['summary']}", "",
                 "board (with hidden colors):"]
        for w in state["words"]:
            lines.append(f"  {w:<14} {state['kinds'][w]}"
                         + ("  [revealed]" if w in state["revealed"] else ""))
        lines.append("")
        for i, t in enumerate(state["log"], 1):
            lines.append(f"Turn {i}: clue \"{t['clue']}\" {t['count']}")
            for w, k in t["guesses"]:
                lines.append(f"  guessed {w} → {k}")
        path = os.path.join(run_dir, "codenames.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return [path]

    def standings(self, labels, results, title="Standings"):
        """Cooperative: rank by average targets found; show clears and assassins."""
        st = {l: {"gp": 0, "found": 0, "clears": 0, "assassin": 0} for l in labels}
        for r in results:
            ex = r.get("extra") or {}
            for label in r["labels"].values():
                if label in st:
                    s = st[label]
                    s["gp"] += 1
                    s["found"] += ex.get("found", 0)
                    s["clears"] += 1 if ex.get("cleared") else 0
                    s["assassin"] += 1 if ex.get("assassin") else 0
        order = sorted(labels, key=lambda l: (-(st[l]["found"] / st[l]["gp"]
                                                if st[l]["gp"] else 0)))
        print(f"\n{BOLD}{title}{RESET}")
        print(f"  {DIM}{'#':>2}  {'competitor':28} {'GP':>3} {'avg found':>9} "
              f"{'clears':>6} {'assassin':>8}{RESET}")
        for i, l in enumerate(order, 1):
            s = st[l]
            avg = s["found"] / s["gp"] if s["gp"] else 0
            print(f"  {i:>2}. {l:28} {s['gp']:>3} {avg:>9.1f} "
                  f"{s['clears']:>6} {s['assassin']:>8}")
