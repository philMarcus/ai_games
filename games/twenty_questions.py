"""Twenty Questions: one model commits a secret, the other deduces it in at
most N yes/no questions (guesses count). The committed secret is an integrity
mechanism — it pins the target so the answerer can't drift mid-game; whether
individual answers were fair is left to reading the saved transcript.

Secret variety: the answerer proposes many candidates and the harness RNG picks
one (the model supplies creativity, the dice supply randomness), excluding
recently used secrets recorded in data/twentyq_recent.json."""

import json
import os
import random
import re

from core.game import Game
from core.term import BOLD, DIM, GREEN, RED, RESET

MIN_CANDIDATES = 8

COMMIT_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {"type": "array", "items": {"type": "string"},
                       "minItems": MIN_CANDIDATES},
        "comment": {"type": "string"},
    },
    "required": ["candidates", "comment"],
}

ASK_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["question", "guess"]},
        "text": {"type": "string"},
        "comment": {"type": "string"},
    },
    "required": ["type", "text", "comment"],
}

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["yes", "no", "unsure"]},
        "comment": {"type": "string"},
    },
    "required": ["answer", "comment"],
}

ANSWERER_SYSTEM = (
    "You are the ANSWERER in a game of 20 Questions against another AI.\n"
    "The game has two phases:\n"
    "1. SETUP (happens once): you propose a list of candidate secrets, and the "
    "game picks ONE of them at random to be THE secret. You do not pick it — "
    "the random draw does.\n"
    "2. ANSWERING: each turn you are shown the chosen secret and the asker's "
    "latest yes/no question about it. Answer truthfully with \"yes\", \"no\", "
    'or "unsure" (only when genuinely unclear). Never reveal the secret itself.\n'
    "The \"comment\" field is PRIVATE — the asker never sees it."
)

ASKER_SYSTEM = (
    "You are the ASKER in a game of 20 Questions against another AI, which has "
    "committed a secret — a common thing (object, animal, person, place...).\n"
    "You have a budget of {limit} questions. Each turn, either ask ONE yes/no "
    "question about the secret, or — when confident — make a guess. A guess "
    "spends one question from the budget; a correct guess wins immediately; if "
    "the budget runs out, you lose.\n"
    "Strategy: binary-search the space (alive? man-made? bigger than a person?) "
    "before guessing.\n"
    "The \"comment\" field is PRIVATE — the answerer never sees it."
)


def norm(s):
    s = re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()
    s = re.sub(r"^(a|an|the)\s+", "", s)
    return s


def guess_matches(guess, secret):
    g, s = norm(guess), norm(secret)
    if not g or not s:
        return False
    return g == s or g in s.split() or s in g.split() or g == s + "s" or s == g + "s"


class TwentyQuestionsGame(Game):
    name = "20q"
    roles = ("answerer", "asker")
    max_rounds_default = 100

    def __init__(self):
        self.limit = 20
        self.rng = random.Random()
        self.recent_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "twentyq_recent.json")

    @classmethod
    def add_args(cls, p):
        p.add_argument("--questions", type=int, default=20,
                       help="Question budget for the asker (default 20)")

    def configure(self, args):
        self.limit = args.questions

    # ── recent-secret memory (variety across sessions) ───────────────────
    def _recent(self):
        try:
            with open(self.recent_path, encoding="utf-8") as f:
                return json.load(f).get("secrets", [])
        except Exception:
            return []

    def _remember(self, secret):
        recent = self._recent()
        recent = [s for s in recent if norm(s) != norm(secret)]
        recent.append(secret)
        os.makedirs(os.path.dirname(self.recent_path), exist_ok=True)
        with open(self.recent_path, "w", encoding="utf-8") as f:
            json.dump({"secrets": recent[-50:]}, f, indent=1)

    # ── state ────────────────────────────────────────────────────────────
    def initial_state(self):
        return {"secret": None, "qa": [], "pending_q": None, "asked": 0,
                "over": False, "won_by": None, "end_summary": None}

    def current_role(self, state):
        if state["secret"] is None or state["pending_q"] is not None:
            return "answerer"
        return "asker"

    def is_over(self, state):
        return state["over"]

    # ── prompts ──────────────────────────────────────────────────────────
    def system_prompt(self, role):
        if role == "answerer":
            return ANSWERER_SYSTEM
        return ASKER_SYSTEM.format(limit=self.limit)

    def _qa_text(self, state):
        if not state["qa"]:
            return "(none yet)"
        return "\n".join(f"{i}. {kind.upper()}: {q} → {a}"
                         for i, (kind, q, a) in enumerate(state["qa"], 1))

    def observation(self, state, role):
        if role == "answerer" and state["secret"] is None:
            recent = self._recent()
            avoid = (f"\nDo NOT propose any of these recently used secrets: "
                     f"{', '.join(recent[-15:])}." if recent else "")
            return (f"Propose {max(20, MIN_CANDIDATES)} DIVERSE candidate secrets — "
                    "a varied mix of animals, objects, foods, places, famous people, "
                    "concepts; common enough to be guessable but not trivial. One "
                    f"will be picked at random as the secret.{avoid}\n"
                    "Your later turns will show you which one was chosen; you will "
                    "then answer the asker's questions about it.\n"
                    "Reply as JSON with the \"candidates\" list.")
        if role == "answerer":
            kind, text = state["pending_q"]
            return (f"The secret for this game is: \"{state['secret']}\" — chosen "
                    "at random by the game from the candidate list YOU proposed "
                    "during setup.\n"
                    f"\nQ&A so far:\n{self._qa_text(state)}\n"
                    f"\nThe asker asks: \"{text}\"\n"
                    'Answer truthfully about that secret as JSON: "answer" must be '
                    '"yes", "no", or "unsure".')
        left = self.limit - state["asked"]
        last = ("\nWARNING: this is your LAST question. A yes/no question now "
                "ends the game and you lose — only a correct guess can win. "
                "Commit to your best guess.\n" if left == 1 else "")
        return (f"Questions remaining: {left} of {self.limit}.\n{last}"
                f"\nQ&A so far:\n{self._qa_text(state)}\n"
                "\nAsk your next yes/no question, or make a guess if confident "
                '(a guess costs one question). Reply as JSON: "type" is '
                '"question" or "guess", "text" is the question or the guessed thing.')

    def action_schema(self, state, role):
        if role == "answerer":
            return COMMIT_SCHEMA if state["secret"] is None else ANSWER_SCHEMA
        return ASK_SCHEMA

    def action_summary(self, action):
        return (action.get("text") or action.get("answer")
                or f"{len(action.get('candidates', []))} candidates" or "?")

    # ── transitions ──────────────────────────────────────────────────────
    def apply(self, state, role, action):
        if role == "answerer" and state["secret"] is None:
            cands = list(dict.fromkeys(
                c.strip() for c in action.get("candidates", []) if str(c).strip()))
            if len(cands) < MIN_CANDIDATES:
                return "illegal", (f"need at least {MIN_CANDIDATES} distinct "
                                   "candidates")
            recent = {norm(s) for s in self._recent()}
            fresh = [c for c in cands if norm(c) not in recent] or cands
            state["secret"] = self.rng.choice(fresh)
            self._remember(state["secret"])
            return "ok", (f"secret committed: {DIM}\"{state['secret']}\"{RESET} "
                          f"(picked from {len(cands)} candidates)")

        if role == "answerer":
            ans = str(action.get("answer", "")).strip().lower()
            if ans not in ("yes", "no", "unsure"):
                return "illegal", 'answer must be "yes", "no", or "unsure"'
            kind, text = state["pending_q"]
            state["qa"].append((kind, text, ans))
            state["pending_q"] = None
            if state["asked"] >= self.limit:
                state["over"] = True
                state["won_by"] = "answerer"
                state["end_summary"] = (f"asker exhausted {self.limit} questions; "
                                        f"the secret was \"{state['secret']}\"")
            return "ok", ans

        # asker
        kind = str(action.get("type", "")).strip().lower()
        text = str(action.get("text", "")).strip()
        if kind not in ("question", "guess") or not text:
            return "illegal", 'need "type" of "question"|"guess" and non-empty "text"'
        state["asked"] += 1
        if kind == "guess":
            if guess_matches(text, state["secret"]):
                state["over"] = True
                state["won_by"] = "asker"
                state["end_summary"] = (f"guessed \"{state['secret']}\" on question "
                                        f"{state['asked']}/{self.limit}")
                return "ok", f"GUESS: \"{text}\" — {GREEN}{BOLD}correct!{RESET}"
            state["qa"].append(("guess", text, "no"))
            if state["asked"] >= self.limit:
                state["over"] = True
                state["won_by"] = "answerer"
                state["end_summary"] = (f"out of questions; the secret was "
                                        f"\"{state['secret']}\"")
            return "ok", f"GUESS: \"{text}\" — {RED}wrong{RESET}"
        state["pending_q"] = (kind, text)
        return "ok", f"Q{state['asked']}: {text}"

    # ── presentation / results ───────────────────────────────────────────
    def render(self, state):
        if state["secret"] is None or state["over"]:
            return ""
        return f"  {DIM}questions used: {state['asked']}/{self.limit}{RESET}"

    def result(self, state, forfeit=None, capped=False):
        if forfeit is not None:
            role, kind = forfeit
            winner = "asker" if role == "answerer" else "answerer"
            why = "ran out of time" if kind == "time" else "illegal action"
            summary = f"forfeit ({role} {why})"
        elif capped:
            winner, summary = None, "round cap reached"
        else:
            winner = state["won_by"]
            summary = state["end_summary"] or "game over"
        return {"winner": winner, "summary": summary,
                "secret": state["secret"], "questions_used": state["asked"],
                "scores": {r: (1.0 if winner == r else 0.0 if winner else 0.5)
                           for r in self.roles}}

    def export(self, state, outcome, run_dir):
        lines = [f"20 Questions: {outcome['labels']['asker']} (asker) vs "
                 f"{outcome['labels']['answerer']} (answerer)",
                 f"secret: {state['secret']}", f"result: {outcome['summary']}", ""]
        for i, (kind, q, a) in enumerate(state["qa"], 1):
            lines.append(f"{i}. {kind.upper()}: {q} → {a}")
        path = os.path.join(run_dir, "qa.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return [path]
