"""Werewolf (Mafia): social deduction for the whole model stable. Hidden roles
— werewolves (who know each other), one seer, villagers. Nights: the wolves
confer privately and choose a victim; the seer inspects someone. Days: the
death is announced, everyone speaks in rotating order for --talk rounds, then
all living players cast SEALED votes; majority is lynched (ties: no lynch),
and the dead player's role is revealed. Village wins when all wolves are dead;
wolves win at parity.

Memory: turns are stateless calls, but every action carries a "notes" field
that is APPENDED to that player's private notebook and shown back to them on
every future turn — continuity of intent without conversation state. The
record contains each player's secret diary next to their public lies.

Spectator vs player display: with no human in the field the terminal shows
everything (roles, night actions, notebooks). The moment a human plays, night
actions are anonymized and roles/notes are hidden — no spoilers."""

import os
import random
import re

from core.competitor import Competitor
from core.game import Game
from core.term import BOLD, CYBER, DIM, GREEN, NAME_PALETTE, RED, RESET, YELLOW


def default_wolves(n_players):
    """Canonical scaling: roughly a quarter of the village are wolves.
    5-6 players: 1-2, 7-9: 2, 10-13: 3, 14-17: 4..."""
    return max(1, (n_players + 2) // 4)


def norm_name(s):
    """Forgiving name matching: models love **bold** and stray punctuation."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

NIGHT_KINDS = ("wolf_msg", "wolf_kill", "see")

SCHEMAS = {
    "wolf_msg": {"type": "object",
                 "properties": {"message": {"type": "string"},
                                "notes": {"type": "string"}},
                 "required": ["message", "notes"]},
    "wolf_kill": {"type": "object",
                  "properties": {"target": {"type": "string"},
                                 "notes": {"type": "string"}},
                  "required": ["target", "notes"]},
    "see": {"type": "object",
            "properties": {"target": {"type": "string"},
                           "notes": {"type": "string"}},
            "required": ["target", "notes"]},
    "speak": {"type": "object",
              "properties": {"speech": {"type": "string"},
                             "notes": {"type": "string"}},
              "required": ["speech", "notes"]},
    "vote": {"type": "object",
             "properties": {"target": {"type": "string"},
                            "notes": {"type": "string"}},
             "required": ["target", "notes"]},
}

SYSTEM = (
    "You are playing WEREWOLF with {n} players: {w} werewolves, 1 seer, and "
    "{v} ordinary villagers. Roles are secret.\n"
    "- The game opens with a quiet first night (NIGHT 0): only the seer acts — "
    "they secretly inspect one player and NO ONE is killed. The werewolves' "
    "first kill comes the following night.\n"
    "- Each NIGHT thereafter the werewolves (who know each other) confer "
    "privately and choose a victim to kill. The seer secretly inspects one "
    "player and learns whether they are a werewolf.\n"
    "- Each DAY the night's death is announced (their role is revealed), all "
    "living players discuss in {talk} rounds of speeches, then everyone "
    "simultaneously casts a SEALED vote. The player with the most votes is "
    "lynched and their role revealed; a tied vote means no one is lynched.\n"
    "- The VILLAGE wins when every werewolf is dead. The WEREWOLVES win when "
    "they equal or outnumber the other survivors.\n"
    "Speeches and votes are public. Werewolves may lie freely — deception is "
    "part of the game. Dead players are out and reveal their role.\n"
    "Every reply includes a \"notes\" field: it is appended to your PRIVATE "
    "notebook and shown back to you on all your future turns. Nobody else ever "
    "sees it — use it to track suspicions, plans, and what you have claimed."
)


class WerewolfGame(Game):
    name = "werewolf"
    roles = ()                  # field game: players come from --models
    max_rounds_default = 2000   # is_over governs
    comment_label = "notebook"  # notes print labeled, on their own line
    num_predict_default = 4096  # speeches + notebooks run long-form

    def __init__(self):
        self.n_wolves = None    # default scales with player count
        self.n_players = 7
        self.talk_rounds = 2
        self.rng = random.Random()
        self.words_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "codenames_words.txt")

    @classmethod
    def add_args(cls, p):
        p.add_argument("--players", type=int, default=7,
                       help="Seats at the table, drawn from the model pool "
                            "(default 7, minimum 5)")
        p.add_argument("--wolves", type=int, default=None,
                       help="Number of werewolves (default scales: ~1 per 4 players)")
        p.add_argument("--talk", type=int, default=2,
                       help="Discussion rounds per day (default 2)")

    def configure(self, args):
        self.n_wolves = args.wolves
        self.n_players = max(5, args.players)
        self.talk_rounds = max(1, args.talk)

    def select_players(self, pool):
        """--models is a POOL: draw n_players seats from it with replacement
        (3 models can fill 7 seats; 15 models yield a random 7). Humans in the
        pool are always seated, once each. Every player gets a friendly table
        name: model base (suffix stripped) + a random word — gemma4-Lantern."""
        comps = list(pool.values())
        humans = [c for c in comps if c.is_human]
        others = [c for c in comps if not c.is_human]
        seats = humans[:self.n_players]
        fill = others or humans
        while len(seats) < self.n_players:
            seats.append(self.rng.choice(fill))
        self.rng.shuffle(seats)
        try:
            with open(self.words_path, encoding="utf-8") as f:
                words = [w.strip() for w in f if w.strip()]
            picks = self.rng.sample(words, len(seats))
        except Exception:
            picks = [str(i) for i in range(1, len(seats) + 1)]
        players = {}
        for c, w in zip(seats, picks):
            base = "human" if c.is_human else c.model.split(":")[0]
            name = f"{base}-{w.title()}"
            players[name] = Competitor(name, c.model, c.think, c.effort,
                                       c.temperature)
        return players

    def set_chain(self, labels):
        if len(labels) < 5:
            import sys
            print("Werewolf needs at least 5 players.")
            sys.exit(1)
        self.roles = tuple(labels)

    # ── state ────────────────────────────────────────────────────────────
    def initial_state(self):
        players = list(self.roles)
        wolves_n = self.n_wolves or default_wolves(len(players))
        shuffled = players[:]
        self.rng.shuffle(shuffled)
        wolves = shuffled[:wolves_n]
        seer = shuffled[wolves_n]
        role_of = {p: ("wolf" if p in wolves else
                       "seer" if p == seer else "villager") for p in players}
        state = {
            "players": players, "role_of": role_of,
            "alive": list(players),
            "phase": "night", "night": 0, "day": 0,
            "queue": [],
            "wolf_chat": [], "wolf_votes": {}, "day_votes": {},
            "seer_log": [],
            "notebooks": {p: [] for p in players},
            "transcript": [],
            "reveal": None,
            "over": False, "team": None,
        }
        state["queue"] = self._night_queue(state)
        return state

    def _wolves(self, state, alive_only=True):
        pool = state["alive"] if alive_only else state["players"]
        return [p for p in pool if state["role_of"][p] == "wolf"]

    def _night_queue(self, state):
        # Night 0 is the quiet opening night: only the seer acts — the wolves
        # already know their allies, so they neither confer nor kill yet.
        q = []
        if state["night"] >= 1:
            wolves = self._wolves(state)
            if len(wolves) >= 2:
                q += [(w, "wolf_msg") for w in wolves]
            q += [(w, "wolf_kill") for w in wolves]
        q += [(p, "see") for p in state["alive"]
              if state["role_of"][p] == "seer"]
        return q

    def _day_queue(self, state):
        # Fresh random speaking order every day — nobody (wolf or villager)
        # systematically anchors or closes the discussion.
        order = list(state["alive"])
        self.rng.shuffle(order)
        q = []
        for _ in range(self.talk_rounds):
            q += [(p, "speak") for p in order]
        q += [(p, "vote") for p in order]
        return q

    def current_role(self, state):
        return state["queue"][0][0]

    def _kind(self, state):
        return state["queue"][0][1]

    def is_over(self, state):
        return state["over"]

    # ── deaths / wins / phase resolution ─────────────────────────────────
    def _kill(self, state, victim, cause):
        state["alive"].remove(victim)
        role = state["role_of"][victim]
        state["transcript"].append(f"{cause}: {victim} is dead. They were a "
                                   f"{role.upper()}.")
        state["queue"] = [(p, k) for p, k in state["queue"] if p != victim]
        state["wolf_votes"].pop(victim, None)
        state["day_votes"].pop(victim, None)

    def _check_win(self, state):
        wolves = len(self._wolves(state))
        others = len(state["alive"]) - wolves
        if wolves == 0:
            state["over"], state["team"] = True, "village"
        elif wolves >= others:
            state["over"], state["team"] = True, "wolves"
        return state["over"]

    def _resolve_night(self, state):
        if state["night"] == 0:
            # The opening night: the seer has peeked, nobody dies.
            state["reveal"] = ("The first night falls — the SEER opens their "
                               "eyes and learns one player's true nature. No "
                               "one is harmed on the opening night.")
        else:
            # Ignore votes against players who died mid-night (e.g. eliminated).
            votes = [t for t in state["wolf_votes"].values()
                     if t in state["alive"]]
            victim = None
            if votes:
                tally = {t: votes.count(t) for t in set(votes)}
                top = max(tally.values())
                victim = self.rng.choice(sorted(t for t, c in tally.items()
                                                if c == top))
            lines = [f"NIGHT {state['night']} ENDS."]
            if victim:
                self._kill(state, victim, f"Night {state['night']}")
                lines.append(f"{victim} was killed in the night — they were a "
                             f"{state['role_of'][victim].upper()}.")
            else:
                lines.append("Nobody died in the night.")
            state["reveal"] = "\n".join(lines)
        state["wolf_chat"], state["wolf_votes"] = [], {}
        if self._check_win(state):
            return
        state["day"] = state["night"] + 1
        state["phase"] = "day"
        state["queue"] = self._day_queue(state)

    def _resolve_day(self, state):
        votes = {v: t for v, t in state["day_votes"].items()
                 if t in state["alive"]}
        tally = {}
        for t in votes.values():
            tally[t] = tally.get(t, 0) + 1
        detail = ", ".join(f"{v}→{t}" for v, t in votes.items())
        lines = [f"DAY {state['day']} VOTE: {detail}"]
        lynched = None
        if tally:
            top = max(tally.values())
            leaders = [t for t, c in tally.items() if c == top]
            if len(leaders) == 1:
                lynched = leaders[0]
        if lynched:
            self._kill(state, lynched, f"Day {state['day']} lynch")
            lines.append(f"{lynched} is lynched — they were a "
                         f"{state['role_of'][lynched].upper()}.")
        else:
            lines.append("The vote is TIED — nobody is lynched.")
        state["transcript"].append(lines[0])
        if not lynched:
            state["transcript"].append("The vote was tied; nobody was lynched.")
        state["day_votes"] = {}
        state["reveal"] = "\n".join(lines)
        if self._check_win(state):
            return
        state["night"] = state["day"]
        state["phase"] = "night"
        state["queue"] = self._night_queue(state)

    def eliminate(self, state, role, kind):
        """Engine hook: a player who can't act is removed; the game goes on."""
        if role in state["alive"]:
            self._kill(state, role, "Forfeit (failed to act)")
        if not self._check_win(state) and not state["queue"]:
            (self._resolve_night if state["phase"] == "night"
             else self._resolve_day)(state)
        return True

    # ── prompts ──────────────────────────────────────────────────────────
    def system_prompt(self, role):
        n = len(self.roles)
        w = self.n_wolves or default_wolves(n)
        return SYSTEM.format(n=n, w=w, v=n - w - 1, talk=self.talk_rounds)

    def _roster(self, state):
        rows = []
        for p in state["players"]:
            if p in state["alive"]:
                rows.append(f"  {p} — alive")
            else:
                rows.append(f"  {p} — DEAD (was {state['role_of'][p]})")
        return "\n".join(rows)

    def _identity(self, state, player):
        me = state["role_of"][player]
        if me == "wolf":
            others = [w + ("" if w in state["alive"] else " (DEAD)")
                      for w in self._wolves(state, alive_only=False)
                      if w != player]
            mates = ", ".join(others) if others else "nobody (you are alone)"
            return (f"You are {player}. You are a WEREWOLF. Your fellow "
                    f"werewolf: {mates}. You are a TEAM: you win or lose "
                    "together, so never vote to lynch a fellow werewolf — but "
                    "the village must never find out who you are.")
        if me == "seer":
            log = ("; ".join(f"night {n}: {t} is "
                             + ("a WEREWOLF" if iw else "NOT a werewolf")
                   for n, t, iw in state["seer_log"]) or "none yet")
            return (f"You are {player}. You are the SEER. Your secret "
                    f"inspection results so far: {log}.")
        return f"You are {player}. You are an ordinary VILLAGER."

    def _notebook(self, state, player):
        notes = state["notebooks"][player]
        if not notes:
            return "(empty so far)"
        return "\n".join(f"  {i}. {n}" for i, n in enumerate(notes, 1))

    def _name_color(self, state, player):
        return NAME_PALETTE[state["players"].index(player) % len(NAME_PALETTE)]

    def _colorize(self, text, state):
        """Tag each player's name in its own bright color (human view only).
        Names are unique hyphenated tokens; longest-first avoids substring
        collisions. RESET/DIM brackets restore the ambient dimming around it."""
        for p in sorted(state["players"], key=len, reverse=True):
            c = self._name_color(state, p)
            text = text.replace(p, f"{RESET}{c}{BOLD}{p}{RESET}{DIM}")
        return text

    def _public(self, state, color=False):
        if not state["transcript"]:
            return "(nothing has happened yet — it is the first night)"
        # A blank line between events, and colored names, for human readers.
        if color:
            events = (self._colorize(e, state) for e in state["transcript"])
            return "\n\n".join(events)
        return "\n".join(state["transcript"])

    def observation(self, state, player):
        kind = self._kind(state)
        human = player in (state.get("_humans") or [])
        living_others = [p for p in state["alive"] if p != player]
        parts = [
            self._identity(state, player),
            f"\nPlayers:\n{self._roster(state)}",
            f"\nPUBLIC EVENTS so far:\n{self._public(state, color=human)}",
            f"\nYOUR PRIVATE NOTEBOOK (only you ever see this):\n"
            f"{self._notebook(state, player)}",
        ]
        if kind == "wolf_msg":
            chat = ("\n".join(f"  {w}: {m}" for w, m in state["wolf_chat"])
                    or "  (no messages yet tonight)")
            if human:
                chat = self._colorize(chat, state)
            parts.append(f"\nNIGHT {state['night']} — private werewolf chat "
                         f"so far:\n{chat}\n\nSend a short private message to "
                         "your fellow werewolf (coordinate your kill). Reply as "
                         'JSON with "message" and "notes".')
        elif kind == "wolf_kill":
            chat = ("\n".join(f"  {w}: {m}" for w, m in state["wolf_chat"])
                    or "  (none)")
            if human:
                chat = self._colorize(chat, state)
            targets = [p for p in living_others
                       if state["role_of"][p] != "wolf"]
            parts.append(f"\nNIGHT {state['night']} — werewolf chat tonight:\n"
                         f"{chat}\n\nChoose tonight's victim. Valid targets: "
                         f"{', '.join(targets)}. Reply as JSON with \"target\" "
                         'and "notes".')
        elif kind == "see":
            parts.append(f"\nNIGHT {state['night']} — choose one player to "
                         f"secretly inspect. Valid targets: "
                         f"{', '.join(living_others)}. You will learn whether "
                         "they are a werewolf. Reply as JSON with \"target\" "
                         'and "notes".')
        elif kind == "speak":
            parts.append(f"\nDAY {state['day']} discussion. Speak to the "
                         "village: accuse, defend, share or fake information — "
                         "everyone will read it. Reply as JSON with \"speech\" "
                         'and "notes".')
        elif kind == "vote":
            parts.append(f"\nDAY {state['day']} — cast your SEALED vote to "
                         f"lynch one player. All votes are revealed together "
                         f"afterwards; most votes is lynched, ties lynch "
                         f"nobody. Valid targets: {', '.join(living_others)}. "
                         'Reply as JSON with "target" and "notes".')
        return "\n".join(parts)

    def action_schema(self, state, role):
        return SCHEMAS[self._kind(state)]

    def action_summary(self, action):
        return (action.get("target") or action.get("speech", "")[:40]
                or action.get("message", "")[:40] or "?")

    def quiet_turn(self, state, role):
        """Anonymize others' night actions when a human is in the game."""
        humans = state.get("_humans") or []
        if not humans:
            return False
        return self._kind(state) in NIGHT_KINDS and role not in humans

    # ── transitions ──────────────────────────────────────────────────────
    def _match_target(self, state, raw, pool):
        """Forgiving: ignores case, markdown (**name**), and stray punctuation."""
        r = norm_name(raw)
        if not r:
            return None
        for p in pool:
            if norm_name(p) == r:
                return p
        return None

    def apply(self, state, player, action):
        kind = self._kind(state)
        spectate = not (state.get("_humans") or [])
        notes = str(action.get("notes", "")).strip()
        living_others = [p for p in state["alive"] if p != player]

        if kind == "wolf_msg":
            msg = str(action.get("message", "")).strip()
            if not msg:
                return "illegal", "empty message"
            state["wolf_chat"].append((player, msg))
            display = (f'(wolf chat) "{msg}"' if spectate
                       else "confers in the dark…")
        elif kind == "wolf_kill":
            pool = [p for p in living_others if state["role_of"][p] != "wolf"]
            target = self._match_target(state, action.get("target"), pool)
            if target is None:
                return "illegal", (f'"{action.get("target")}" is not a valid '
                                   f"victim — choose one of: {', '.join(pool)}")
            state["wolf_votes"][player] = target
            display = (f"marks {target} for death" if spectate
                       else "chooses in the dark…")
        elif kind == "see":
            target = self._match_target(state, action.get("target"),
                                        living_others)
            if target is None:
                return "illegal", (f'"{action.get("target")}" is not a valid '
                                   f"target — choose one of: "
                                   f"{', '.join(living_others)}")
            is_wolf = state["role_of"][target] == "wolf"
            state["seer_log"].append((state["night"], target, is_wolf))
            display = (f"inspects {target} — "
                       + ("a WEREWOLF" if is_wolf else "not a wolf")
                       if spectate else "peers into the night…")
        elif kind == "speak":
            speech = str(action.get("speech", "")).strip()
            if not speech:
                return "illegal", "empty speech"
            state["transcript"].append(f"Day {state['day']}, {player} says: "
                                       f"\"{speech}\"")
            display = f'says: "{speech}"'
        elif kind == "vote":
            target = self._match_target(state, action.get("target"),
                                        living_others)
            if target is None:
                return "illegal", (f'"{action.get("target")}" is not a valid '
                                   f"vote — choose one of: "
                                   f"{', '.join(living_others)}")
            state["day_votes"][player] = target
            display = "seals their vote"
        else:
            return "illegal", f"unknown action kind {kind}"

        if notes:
            tag = (f"night {state['night']}" if kind in NIGHT_KINDS
                   else f"day {state['day']}")
            state["notebooks"][player].append(f"({tag}) {notes}")
        self._last_notes = notes if spectate else ""

        state["queue"].pop(0)
        if not state["queue"]:
            (self._resolve_night if state["phase"] == "night"
             else self._resolve_day)(state)
        return "ok", display

    # ── presentation / results ───────────────────────────────────────────
    def comment_of(self, action):
        # apply() stashes the notes (spectator mode only) for the engine's
        # comment display, since this hook can't see state.
        return getattr(self, "_last_notes", "")

    def render(self, state):
        out = []
        if state["reveal"]:
            out.append(f"\n  {BOLD}{YELLOW}{state['reveal']}{RESET}")
            state["reveal"] = None
        if state["over"]:
            return "\n".join(out)
        spectate = not (state.get("_humans") or [])

        def name(p):
            mark = "" if p in state["alive"] else f"{DIM}✗ "
            end = "" if p in state["alive"] else f"{RESET}"
            return f"{mark}{p}{end}"

        phase = (f"night {state['night']}" if state["phase"] == "night"
                 else f"day {state['day']}")
        if spectate:
            groups = []
            for kind, color in (("wolf", RED), ("seer", CYBER),
                                ("villager", "")):
                members = [name(p) for p in state["players"]
                           if state["role_of"][p] == kind]
                label = {"wolf": "WOLVES", "seer": "SEER",
                         "villager": "VILLAGERS"}[kind]
                groups.append(f"{color}{label}:{RESET} " + ", ".join(members))
            out.append(f"\n  {DIM}[{phase}]{RESET}  " + "   ".join(groups))
        else:
            alive = ", ".join(p for p in state["alive"])
            dead = ", ".join(f"✗ {p} (was {state['role_of'][p]})"
                             for p in state["players"]
                             if p not in state["alive"])
            line = f"\n  {DIM}[{phase}]{RESET}  alive: {alive}"
            if dead:
                line += f"   {DIM}dead: {dead}{RESET}"
            out.append(line)
        return "\n".join(out)

    def result(self, state, forfeit=None, capped=False):
        team = state["team"]
        roles = state["role_of"]
        if capped and team is None:
            summary = "game called at the round cap (no winner)"
        elif team == "village":
            summary = "the VILLAGE wins — all werewolves are dead"
        elif team == "wolves":
            summary = "the WEREWOLVES win — they reached parity"
        else:
            summary = "game over"
        won = {p: (team == "wolves") == (roles[p] == "wolf") and team is not None
               for p in state["players"]}
        return {"winner": None, "summary": summary,
                "no_winner_banner": summary,
                "scores": {p: (1.0 if won[p] else 0.0) for p in state["players"]},
                "extra": {"team": team, "roles": roles,
                          "survivors": list(state["alive"]),
                          "days": state["day"],
                          "won": won}}

    def export(self, state, outcome, run_dir):
        import os
        lines = [f"WEREWOLF — {outcome['summary']}", "", "Cast:"]
        for p in state["players"]:
            status = "survived" if p in state["alive"] else "died"
            lines.append(f"  {p}: {state['role_of'][p]} ({status})")
        lines += ["", "The story:"] + state["transcript"]
        lines += ["", "The private notebooks:"]
        for p in state["players"]:
            lines.append(f"\n--- {p} ({state['role_of'][p]}) ---")
            lines += [f"  {n}" for n in state["notebooks"][p]] or ["  (empty)"]
        path = os.path.join(run_dir, "story.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return [path]

    def standings(self, labels, results, title="Standings"):
        """Aggregated by underlying MODEL — table names change every game."""
        st = {}
        for r in results:
            ex = r.get("extra") or {}
            roles, won = ex.get("roles", {}), ex.get("won", {})
            models = r.get("models", {})
            for p in roles:
                key = models.get(p, p)
                s = st.setdefault(key, {"gp": 0, "wins": 0, "wolf_gp": 0,
                                        "wolf_w": 0, "vill_w": 0, "survived": 0})
                s["gp"] += 1
                if roles[p] == "wolf":
                    s["wolf_gp"] += 1
                if won.get(p):
                    s["wins"] += 1
                    if roles[p] == "wolf":
                        s["wolf_w"] += 1
                    else:
                        s["vill_w"] += 1
                if p in ex.get("survivors", []):
                    s["survived"] += 1
        order = sorted(st, key=lambda l: -(st[l]["wins"] / st[l]["gp"]
                                           if st[l]["gp"] else 0))
        print(f"\n{BOLD}{title}{RESET}")
        print(f"  {DIM}{'#':>2}  {'competitor':28} {'GP':>3} {'W':>3} "
              f"{'win%':>5} {'as-wolf':>8} {'as-vill':>8} {'lived':>5}{RESET}")
        for i, l in enumerate(order, 1):
            s = st[l]
            pct = 100 * s["wins"] / s["gp"] if s["gp"] else 0
            wolf = f"{s['wolf_w']}/{s['wolf_gp']}"
            vill = f"{s['vill_w']}/{s['gp'] - s['wolf_gp']}"
            print(f"  {i:>2}. {l:28} {s['gp']:>3} {s['wins']:>3} "
                  f"{pct:>4.0f}% {wolf:>8} {vill:>8} {s['survived']:>5}")
