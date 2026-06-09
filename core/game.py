"""The Game interface: a game plugs its rules, hidden information, and scoring
into the shared engine (Ollama, roster, clock, retries/forfeit, tournaments,
records). All turn structures run through the same sequential loop — simultaneous
games (e.g. prisoner's dilemma) hold the first mover's action hidden inside state
until the second mover commits, which works because observation() controls what
each role sees."""

import json

from core.term import BOLD, DIM, RESET


class Game:
    name = "game"
    roles = ("p1", "p2")        # role names; the CLI grows a --<role> flag for each
    max_rounds_default = 1000   # safety cap on successful actions per game

    # ── configuration ────────────────────────────────────────────────────
    @classmethod
    def add_args(cls, parser):
        """Add game-specific CLI flags."""

    def configure(self, args):
        """Stash game options from parsed args (called once before play)."""

    def close(self):
        """Release external resources (e.g. an engine process) at exit."""

    # ── rules / state ────────────────────────────────────────────────────
    def initial_state(self):
        raise NotImplementedError

    def current_role(self, state):
        """Which role acts now."""
        raise NotImplementedError

    def system_prompt(self, role):
        raise NotImplementedError

    def observation(self, state, role):
        """The prompt text this role sees — respecting hidden information."""
        raise NotImplementedError

    def action_schema(self, role):
        """JSON schema for this role's action (Ollama structured output)."""
        raise NotImplementedError

    def action_summary(self, action):
        """Short string naming an action (for already-tried feedback lists)."""
        return json.dumps(action, ensure_ascii=False)

    def comment_of(self, action):
        """The private commentary attached to an action, if any."""
        return str(action.get("comment", "")).strip()

    def apply(self, state, role, action):
        """Validate + transition. Returns ("ok", display_str) on success or
        ("illegal", reason) — the reason is fed back to the model on retry."""
        raise NotImplementedError

    def is_over(self, state):
        raise NotImplementedError

    # ── presentation / results ───────────────────────────────────────────
    def render(self, state):
        """Human terminal display after each action ('' for none)."""
        return ""

    def result(self, state, forfeit=None, capped=False):
        """Outcome dict once the game ends. `forfeit` is (role, kind) when a
        player failed to act (kind: 'illegal'|'time'); `capped` means the
        round cap was hit. Must include: winner (role name or None) and
        summary (one-line human description). May add game-specific fields."""
        raise NotImplementedError

    def export(self, state, outcome, run_dir):
        """Write game-native transcript(s) (PGN/SGF/log) into the run dir.
        Returns list of paths written."""
        return []

    # ── standings (game-defined; W/L/D table is just the default) ────────
    def standings(self, labels, results, title="Standings"):
        """Print a leaderboard from per-game result records. Default: W/L/D
        points table (suits win/lose/draw games). Point-scored or cooperative
        games override this."""
        table = {l: {"w": 0, "l": 0, "d": 0, "pts": 0.0, "gp": 0, "ff": 0}
                 for l in labels}
        for r in results:
            ls = list(r["labels"].values())
            winner = r.get("winner")
            for l in ls:
                if l in table:
                    table[l]["gp"] += 1
            if winner is None:
                for l in ls:
                    table[l]["d"] += 1
                    table[l]["pts"] += 0.5
            else:
                table[winner]["w"] += 1
                table[winner]["pts"] += 1
                for l in ls:
                    if l != winner:
                        table[l]["l"] += 1
                        if r.get("forfeit"):
                            table[l]["ff"] += 1
        order = sorted(labels, key=lambda l: (-table[l]["pts"], -table[l]["w"]))
        print(f"\n{BOLD}{title}{RESET}")
        print(f"  {DIM}{'#':>2}  {'competitor':28} {'pts':>5} {'W':>3} {'L':>3} "
              f"{'D':>3} {'GP':>3}  {'win%':>5}  ff{RESET}")
        for i, l in enumerate(order, 1):
            t = table[l]
            winpct = (100 * t["pts"] / t["gp"]) if t["gp"] else 0
            print(f"  {i:>2}. {l:28} {t['pts']:>5.1f} {t['w']:>3} {t['l']:>3} "
                  f"{t['d']:>3} {t['gp']:>3}  {winpct:>4.0f}%  {t['ff'] or '':>2}")
