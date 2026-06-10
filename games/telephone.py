"""Telephone: the kids' game, played by the whole model stable. The first model
composes a seed text; every model after it is shown only the previous model's
output and must repeat it EXACTLY. Outputs propagate, so errors compound —
classic telephone. We count mutations (any change after whitespace
normalization) over a fixed number of steps, or stop at the first one.

Measures pure transcription fidelity / instruction-following: the urge to
"fix" typos, normalize quotes, drop words, or editorialize is exactly what
gets caught here. Standings rank models by fidelity across games."""

import difflib
import os
import random
import re

from core.game import Game
from core.term import BOLD, DIM, GREEN, RED, RESET, YELLOW

SEED_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "comment": {"type": "string"},
    },
    "required": ["text"],
}

LENGTH_SPECS = {
    "phrase": "a short phrase of roughly 3-8 words",
    "sentence": "one complete sentence of roughly 10-25 words",
    "paragraph": "a short paragraph of 3-5 sentences",
}

SEED_SYSTEM = (
    "You are starting a game of telephone between AI models. Compose an "
    "original, interesting {spec} — vivid and specific beats bland. It will be "
    "passed down a chain of models, each repeating it to the next.\n"
    'Reply with ONLY a JSON object {{"text": "<your composition>"}}. The '
    '"text" field must contain the composition itself and NOTHING else.'
)

REPEAT_SYSTEM = (
    "You are one link in a chain of AI models playing telephone. You are given "
    "a text. Your ONLY job is to repeat it EXACTLY — every word, every "
    "punctuation mark, every capitalization, unchanged. Do not correct "
    "mistakes, do not improve wording, do not add or remove anything.\n"
    'Reply with ONLY a JSON object {"text": "<the exact text>"}. The "text" '
    "field must contain the repeated text and NOTHING else."
)


def norm(s):
    """Whitespace-insensitive form: runs of whitespace collapse to one space."""
    return re.sub(r"\s+", " ", str(s)).strip()


def similarity(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


class TelephoneGame(Game):
    name = "telephone"
    roles = ()                  # chain game: competitors come from --models
    max_rounds_default = 1000   # is_over governs

    def __init__(self):
        self.length = "sentence"
        self.steps = 50
        self.stop_on_mutation = False
        self.rng = random.Random()
        self.words_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "codenames_words.txt")

    @classmethod
    def add_args(cls, p):
        p.add_argument("--length", choices=list(LENGTH_SPECS), default="sentence",
                       help="What the first model composes (default sentence)")
        p.add_argument("--steps", type=int, default=50,
                       help="Repeat steps after the seed (default 50)")
        p.add_argument("--stop-on-mutation", action="store_true",
                       help="End the game at the first mutation instead")

    def configure(self, args):
        self.length = args.length
        self.steps = args.steps
        self.stop_on_mutation = args.stop_on_mutation

    def set_chain(self, labels):
        """Called by the CLI with the chain order; roles ARE the labels."""
        self.roles = tuple(labels)

    # ── state ────────────────────────────────────────────────────────────
    def initial_state(self):
        try:
            with open(self.words_path, encoding="utf-8") as f:
                pool = [w.strip() for w in f if w.strip()]
            hints = ", ".join(self.rng.sample(pool, 2)).lower()
        except Exception:
            hints = "anything you like"
        return {"texts": [],            # texts[0] = seed, then one per repeat
                "steps": [],            # per repeat: {label, mutated, similarity}
                "mutations": 0, "first_mutation_step": None,
                "hints": hints}

    def current_role(self, state):
        return self.roles[len(state["texts"]) % len(self.roles)]

    def is_over(self, state):
        repeats = max(0, len(state["texts"]) - 1)
        if self.stop_on_mutation and state["mutations"]:
            return True
        return repeats >= self.steps

    # ── prompts ──────────────────────────────────────────────────────────
    def system_prompt(self, role):
        # The seed turn is texts==[], but system_prompt has no state; both
        # prompts are delivered via observation instead. Use a neutral shared
        # system and put the real instruction in the observation.
        return ("You are playing a game of telephone between AI models. Follow "
                "the instruction in the message exactly. Reply with ONLY the "
                "requested JSON object.")

    def observation(self, state, role):
        if not state["texts"]:
            return (SEED_SYSTEM.format(spec=LENGTH_SPECS[self.length])
                    + f"\nFor inspiration (optional): {state['hints']}.")
        return (REPEAT_SYSTEM
                + "\n\nThe text to repeat exactly:\n" + state["texts"][-1])

    def action_schema(self, state, role):
        return SEED_SCHEMA

    def action_summary(self, action):
        return norm(action.get("text", ""))[:40] or "?"

    def comment_of(self, action):
        return str(action.get("comment", "")).strip()

    # ── transitions ──────────────────────────────────────────────────────
    def apply(self, state, role, action):
        text = str(action.get("text", ""))
        if not norm(text):
            return "illegal", "empty text"
        if not state["texts"]:
            state["texts"].append(text)
            return "ok", f"seed ({len(norm(text).split())} words)"
        prev = state["texts"][-1]
        mutated = norm(text) != norm(prev)
        sim = similarity(prev, text)
        state["texts"].append(text)
        step_no = len(state["texts"]) - 1
        state["steps"].append({"label": role, "mutated": mutated,
                               "similarity": round(sim, 4)})
        if mutated:
            state["mutations"] += 1
            if state["first_mutation_step"] is None:
                state["first_mutation_step"] = step_no
            return "ok", (f"step {step_no}: {RED}MUTATED{RESET} "
                          f"{DIM}(similarity {sim:.2f}){RESET}")
        return "ok", f"step {step_no}: {GREEN}faithful{RESET}"

    # ── presentation / results ───────────────────────────────────────────
    def render(self, state):
        if not state["texts"]:
            return ""
        text = norm(state["texts"][-1])
        if len(text) > 240:
            text = text[:240] + "…"
        tally = (f"  {DIM}mutations so far: {state['mutations']}"
                 + (f" (first at step {state['first_mutation_step']})"
                    if state["first_mutation_step"] else "") + f"{RESET}")
        return f"  {DIM}“{RESET}{text}{DIM}”{RESET}\n{tally}"

    def result(self, state, forfeit=None, capped=False):
        repeats = max(0, len(state["texts"]) - 1)
        per_label = {}
        for s in state["steps"]:
            d = per_label.setdefault(s["label"], {"repeats": 0, "mutations": 0,
                                                  "sims": []})
            d["repeats"] += 1
            d["mutations"] += 1 if s["mutated"] else 0
            d["sims"].append(s["similarity"])
        for d in per_label.values():
            d["avg_similarity"] = round(sum(d["sims"]) / len(d["sims"]), 4)
            del d["sims"]
        if forfeit is not None:
            role, kind = forfeit
            why = "ran out of time" if kind == "time" else "couldn't produce text"
            summary = (f"forfeit ({role} {why}) after {repeats} of "
                       f"{self.steps} steps; {state['mutations']} mutations")
        elif self.stop_on_mutation and state["mutations"]:
            summary = f"first mutation at step {state['first_mutation_step']}"
        else:
            first = (f", first at step {state['first_mutation_step']}"
                     if state["first_mutation_step"] else "")
            summary = (f"{state['mutations']} mutation(s) in {repeats} "
                       f"steps{first}")
        return {"winner": None, "summary": summary,
                "no_winner_banner": ("survived unchanged"
                                     if not state["mutations"] else summary),
                "scores": {l: float(d["mutations"]) for l, d in per_label.items()},
                "extra": {"per_label": per_label, "repeats": repeats,
                          "mutations": state["mutations"],
                          "first_mutation_step": state["first_mutation_step"],
                          "seed": norm(state["texts"][0]) if state["texts"] else None,
                          "final": norm(state["texts"][-1]) if state["texts"] else None}}

    def export(self, state, outcome, run_dir):
        lines = [f"Telephone — {outcome['summary']}", ""]
        if state["texts"]:
            lines.append(f"SEED ({self.current_seed_label()}): {state['texts'][0]}")
        for i, s in enumerate(state["steps"], 1):
            if s["mutated"]:
                lines.append(f"step {i} [{s['label']}] MUTATED "
                             f"(sim {s['similarity']}):")
                lines.append(f"  {state['texts'][i]}")
            else:
                lines.append(f"step {i} [{s['label']}] unchanged")
        path = os.path.join(run_dir, "telephone.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return [path]

    def current_seed_label(self):
        return self.roles[0] if self.roles else "?"

    def standings(self, labels, results, title="Standings"):
        """Fidelity leaderboard: fewest mutations per repeat across games."""
        st = {l: {"repeats": 0, "mutations": 0, "sim": 0.0} for l in labels}
        for r in results:
            per = (r.get("extra") or {}).get("per_label", {})
            for label, d in per.items():
                if label in st:
                    s = st[label]
                    s["repeats"] += d["repeats"]
                    s["mutations"] += d["mutations"]
                    s["sim"] += d["avg_similarity"] * d["repeats"]
        def fidelity(l):
            s = st[l]
            return 1 - (s["mutations"] / s["repeats"]) if s["repeats"] else 0
        order = sorted(labels, key=lambda l: -fidelity(l))
        print(f"\n{BOLD}{title}{RESET}")
        print(f"  {DIM}{'#':>2}  {'competitor':28} {'repeats':>7} {'mut':>4} "
              f"{'fidelity':>8} {'avg sim':>8}{RESET}")
        for i, l in enumerate(order, 1):
            s = st[l]
            avg_sim = s["sim"] / s["repeats"] if s["repeats"] else 0
            print(f"  {i:>2}. {l:28} {s['repeats']:>7} {s['mutations']:>4} "
                  f"{100 * fidelity(l):>7.0f}% {avg_sim:>8.3f}")
