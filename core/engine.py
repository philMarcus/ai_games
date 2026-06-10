"""The game-agnostic engine: the per-turn action loop (structured output →
validate → retry with accumulated feedback → forfeit), the per-move clock, game
playback with terminal display, records-by-default, head-to-head matches, and
resumable round-robin tournaments."""

import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime

from core.term import BOLD, CYBER, DIM, GREEN, RED, RESET, WHITE, YELLOW

ROLE_COLORS = (WHITE, CYBER, YELLOW, GREEN)  # cycled per role for turn headers


# ──────────────────────────────────────────────────────────────────────────
# Structured-output helpers
# ──────────────────────────────────────────────────────────────────────────

def extract_json(raw):
    """Best-effort parse of a JSON object out of model content."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def atomic_write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def slug(s):
    return re.sub(r"[^A-Za-z0-9.]+", "-", str(s)).strip("-")


# ──────────────────────────────────────────────────────────────────────────
# One turn: get a legal action from a competitor (retries → forfeit)
# ──────────────────────────────────────────────────────────────────────────

def strip_comment_schema(schema):
    """--no-comment: remove the private-comment field from an action schema."""
    s = json.loads(json.dumps(schema))
    s.get("properties", {}).pop("comment", None)
    if "required" in s:
        s["required"] = [f for f in s["required"] if f != "comment"]
    return s


def prompt_human_field(name, spec, required):
    """Read one schema field from the keyboard. Returns the converted value,
    or None for an optional field left empty. Raises EOFError if input ends."""
    hints = []
    if "enum" in spec:
        hints.append("/".join(str(e) for e in spec["enum"]))
    elif spec.get("type") == "integer":
        hints.append("number")
    elif spec.get("type") == "array":
        hints.append("comma-separated")
    if not required:
        hints.append("optional")
    hint = f" ({'; '.join(hints)})" if hints else ""
    while True:
        raw = input(f"  {name}{hint}> ").strip()
        if not raw:
            if required:
                print("  (required)")
                continue
            return None
        if spec.get("type") == "integer":
            try:
                return int(raw)
            except ValueError:
                print("  (enter a whole number)")
                continue
        if spec.get("type") == "array":
            return [s.strip() for s in raw.split(",") if s.strip()]
        if "enum" in spec:
            for e in spec["enum"]:
                if raw.lower() == str(e).lower():
                    return e
            print(f"  (one of: {', '.join(str(e) for e in spec['enum'])})")
            continue
        return raw


def human_turn(game, state, role, comp, schema, opts, events):
    """A human plays this turn from the keyboard, prompted per schema field.
    Unlimited retries on illegal actions (a typo shouldn't forfeit), but the
    per-move clock still applies."""
    deadline = (time.time() + opts["move_time"]) if opts["move_time"] else None
    print(f"{DIM}{game.observation(state, role)}{RESET}")
    if deadline:
        print(f"{YELLOW}  (you have {opts['move_time']:g}s on the clock){RESET}")
    while True:
        if deadline is not None and time.time() >= deadline:
            return "forfeit", "time", None
        t0 = time.time()
        action = {}
        try:
            for name, spec in schema.get("properties", {}).items():
                required = name in schema.get("required", []) and name != "comment"
                val = prompt_human_field(name, spec, required)
                if val is not None:
                    action[name] = val
        except EOFError:
            return "forfeit", "illegal", None
        elapsed = time.time() - t0
        if deadline is not None and time.time() >= deadline:
            print(f"{RED}  ✗ too slow — flag fall{RESET}")
            return "forfeit", "time", None
        verdict, info = game.apply(state, role, action)
        if verdict == "ok":
            events.append({"type": "action", "role": role, "label": comp.label,
                           "action": action, "display": info,
                           "elapsed": round(elapsed, 2)})
            comment = "" if opts.get("no_comment") else game.comment_of(action)
            return "ok", info, comment
        print(f"{RED}  ! illegal: {game.action_summary(action)} ({info}) "
              f"— try again{RESET}")
        events.append({"type": "illegal", "role": role, "label": comp.label,
                       "action": action, "reason": info,
                       "elapsed": round(elapsed, 2)})


def take_turn(client, game, state, role, comp, opts, events):
    """Run one role's turn. On success the action is applied to state.
    Returns ("ok", display, comment) or ("forfeit", kind) with kind
    'illegal'|'time'. Appends attempt records to `events`."""
    schema = game.action_schema(state, role)
    if opts.get("no_comment"):
        schema = strip_comment_schema(schema)
    if comp.is_human:
        return human_turn(game, state, role, comp, schema, opts, events)

    system = game.system_prompt(role)
    if opts["move_time"]:
        system += (f"\nTIME CONTROL: you have at most {opts['move_time']} seconds to "
                   "act each turn. If you do not produce a legal action in time, you "
                   "lose. Reason efficiently and answer before your time runs out.")
    deadline = (time.time() + opts["move_time"]) if opts["move_time"] else None

    think_open = [False]
    on_think = None
    if opts["show_think"] and comp.think:
        def on_think(tok):
            if not think_open[0]:
                sys.stdout.write(f"{DIM}  (thinking) ")
                think_open[0] = True
            sys.stdout.write(tok)
            sys.stdout.flush()

    feedback = None
    tried = []   # illegal actions attempted this turn, fed back so it stops repeating them
    out_of_time = False
    retries = opts["retries"]
    for attempt in range(retries + 1):
        if deadline is not None and time.time() >= deadline:
            out_of_time = True
            break
        obs = game.observation(state, role)
        if feedback:
            obs = ("FEEDBACK ON YOUR PREVIOUS REPLY (this same turn — it was "
                   f"rejected; the state shown below is unchanged):\n{feedback}"
                   f"\n\n{obs}")
        think_open[0] = False
        t0 = time.time()
        raw, think_chars, timed_out = client.chat(
            comp.model,
            [{"role": "system", "content": system},
             {"role": "user", "content": obs}],
            schema=schema, temperature=comp.temperature,
            num_predict=opts["num_predict"], think=comp.think,
            think_effort=comp.effort, deadline=deadline, on_think=on_think)
        elapsed = time.time() - t0
        if think_open[0]:
            sys.stdout.write(f"{RESET}\n")
            sys.stdout.flush()

        more = attempt < retries
        tail = f" — retrying ({attempt + 1}/{retries})" if more else " — no retries left"
        action = extract_json(raw)

        if not isinstance(action, dict):
            if timed_out:
                out_of_time = True
                break  # out of time mid-generation → time forfeit
            snippet = " ".join(raw.split())[:160]
            if think_chars and not raw.strip():
                kind = "think_exhausted"
                sys.stdout.write(f"{RED}  ! used its entire token budget thinking "
                                 f"({think_chars} chars) and never answered{tail}{RESET}\n")
                feedback = ("You spent your whole response thinking and never output an "
                            "action. Think briefly, then immediately output ONLY the "
                            "required JSON.")
            else:
                kind = "unparseable"
                sys.stdout.write(f"{RED}  ! no parseable action in reply (bad JSON)"
                                 f"{tail}{RESET}\n")
                sys.stdout.write(f"{DIM}    got: {snippet or '(empty response)'}{RESET}\n")
                feedback = ("Your previous reply was not the valid JSON object that was "
                            "asked for. Reply with ONLY that JSON.")
            events.append({"type": "bad_json", "kind": kind, "role": role,
                           "label": comp.label, "think_chars": think_chars,
                           "snippet": snippet, "elapsed": round(elapsed, 2)})
            continue

        verdict, info = game.apply(state, role, action)
        if verdict == "ok":
            events.append({"type": "action", "role": role, "label": comp.label,
                           "action": action, "display": info,
                           "elapsed": round(elapsed, 2)})
            comment = "" if opts.get("no_comment") else game.comment_of(action)
            return "ok", info, comment

        # illegal
        if timed_out:
            out_of_time = True
            break
        name = game.action_summary(action)
        tried.append(name)
        already = ", ".join(f'"{t}"' for t in tried)
        feedback = (f"Earlier on THIS SAME turn you already tried {already}; each was "
                    f"rejected as illegal: {info}. Do not propose any of those again; "
                    "choose a different, legal action.")
        sys.stdout.write(f"{RED}  ! illegal: {name} ({info}){tail}{RESET}\n")
        comment = "" if opts.get("no_comment") else game.comment_of(action)
        if comment:
            sys.stdout.write(f"{DIM}    “{comment}”{RESET}\n")
        events.append({"type": "illegal", "role": role, "label": comp.label,
                       "action": action, "reason": info,
                       "elapsed": round(elapsed, 2)})

    if opts["move_time"] and (out_of_time or
                              (deadline is not None and time.time() >= deadline)):
        return "forfeit", "time", None
    return "forfeit", "illegal", None


# ──────────────────────────────────────────────────────────────────────────
# Playing one game (with records-by-default)
# ──────────────────────────────────────────────────────────────────────────

def play_game(client, game, assignment, opts, header=""):
    """Play one game. `assignment` maps role -> Competitor. Returns the outcome
    record (also written to a run dir unless opts['record'] is False)."""
    state = game.initial_state()
    events = []
    custom_colors = getattr(game, "role_colors", {})
    role_color = {r: custom_colors.get(r, ROLE_COLORS[i % len(ROLE_COLORS)])
                  for i, r in enumerate(game.roles)}
    labels = {r: assignment[r].label for r in game.roles}
    forfeit = None
    started = time.time()

    if header:
        print(f"\n{BOLD}{header}{RESET}")
    print(f"{DIM}{'─' * 60}{RESET}")
    for r in game.roles:
        c = assignment[r]
        who = c.label if r == c.label else f"{r.capitalize()}{RESET}: {c.label}"
        print(f"  {role_color[r]}{BOLD}{who}{RESET} {DIM}[{c.desc()}]{RESET}")
    first = game.render(state)
    if first:
        print(first)

    turns = 0
    max_rounds = opts["max_rounds"]
    while not game.is_over(state) and turns < max_rounds:
        role = game.current_role(state)
        comp = assignment[role]
        turn_title = (comp.label if role == comp.label
                      else f"{role.capitalize()} — {comp.label}")
        print(f"\n{BOLD}{role_color[role]}{turn_title}{RESET}")

        t0 = time.time()
        got = take_turn(client, game, state, role, comp, opts, events)
        elapsed = time.time() - t0

        if got[0] == "forfeit":
            kind = got[1]
            forfeit = (role, kind)
            reason = ("ran out of time" if kind == "time"
                      else "failed to produce a legal action")
            print(f"{RED}{BOLD}  ✗ {role.capitalize()} ({comp.label}) {reason} "
                  f"({elapsed:.0f}s) — forfeits.{RESET}")
            break

        _, display, comment = got
        turns += 1
        clock = (f"{elapsed:.1f}s/{opts['move_time']}s" if opts["move_time"]
                 else f"{elapsed:.1f}s")
        over = opts["move_time"] and elapsed > opts["move_time"]
        print(f"  {BOLD}{display}{RESET}   {RED if over else DIM}({clock}){RESET}")
        if comment:
            ccol = WHITE if role == game.roles[0] else DIM
            print(f"  {ccol}“{comment}”{RESET}")
        board = game.render(state)
        if board:
            print(board)
        if opts["delay"]:
            time.sleep(opts["delay"])

    capped = not game.is_over(state) and forfeit is None
    outcome = game.result(state, forfeit=forfeit, capped=capped)

    # Enrich with the generic record fields.
    outcome.update({
        "game": game.name,
        "labels": labels,
        "models": {r: assignment[r].model for r in game.roles},
        "competitors": {r: assignment[r].to_cfg() for r in game.roles},
        "winner": labels.get(outcome.get("winner")) if outcome.get("winner") else None,
        "turns": turns,
        "forfeit": ({"role": forfeit[0], "label": labels[forfeit[0]],
                     "kind": forfeit[1]} if forfeit else None),
        "duration_s": round(time.time() - started, 1),
        "stats": generic_stats(game, labels, events),
        "when": datetime.now().isoformat(timespec="seconds"),
    })

    winner = outcome["winner"]
    banner = (f"{GREEN}{BOLD}{winner} wins{RESET}" if winner
              else f"{YELLOW}{BOLD}{outcome.get('no_winner_banner', 'draw')}{RESET}")
    print(f"\n  {banner}  {DIM}— {outcome.get('summary', '')}, {turns} turns{RESET}")

    if opts.get("record", True):
        run_dir = save_run(game, state, outcome, events, opts)
        outcome["run_dir"] = run_dir
        print(f"  {DIM}recorded: {run_dir}{RESET}")
    return outcome


def generic_stats(game, labels, events):
    """Cross-game reliability stats per label: illegal/bad-json counts, turn of
    first illegal, mean action time."""
    stats = {l: {"actions": 0, "illegal": 0, "bad_json": 0,
                 "first_illegal_turn": None, "mean_time_s": None, "_times": []}
             for l in labels.values()}
    turn = 0
    for e in events:
        s = stats.get(e["label"])
        if s is None:
            continue
        if e["type"] == "action":
            turn += 1
            s["actions"] += 1
            s["_times"].append(e["elapsed"])
        elif e["type"] == "illegal":
            s["illegal"] += 1
            if s["first_illegal_turn"] is None:
                s["first_illegal_turn"] = turn + 1
        elif e["type"] == "bad_json":
            s["bad_json"] += 1
    for s in stats.values():
        if s["_times"]:
            s["mean_time_s"] = round(sum(s["_times"]) / len(s["_times"]), 2)
        del s["_times"]
    return stats


def save_run(game, state, outcome, events, opts):
    """Records by default: every game writes a self-contained record under runs/."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    labels = list(dict.fromkeys(outcome["labels"].values()))
    names = "_vs_".join(slug(l) for l in labels)
    if len(names) > 60:   # long chains would blow past Windows path limits
        names = f"{slug(labels[0])}_and_{len(labels) - 1}_others"
    run_dir = os.path.join(opts["runs_dir"], f"{ts}_{game.name}_{names}")
    n, base = 2, run_dir
    while os.path.exists(run_dir):
        run_dir = f"{base}-{n}"
        n += 1
    os.makedirs(run_dir, exist_ok=True)
    record = dict(outcome)
    record["events"] = events
    atomic_write_json(os.path.join(run_dir, "record.json"), record)
    try:
        game.export(state, outcome, run_dir)
    except Exception as e:
        print(f"{YELLOW}  (transcript export failed: {e}){RESET}")
    return run_dir


# ──────────────────────────────────────────────────────────────────────────
# Head-to-head match
# ──────────────────────────────────────────────────────────────────────────

def run_match(client, game, comp_a, comp_b, games, opts):
    """N games between two competitors, swapping the first role each game."""
    results = []
    for gnum in range(1, games + 1):
        pair = (comp_a, comp_b) if gnum % 2 == 1 else (comp_b, comp_a)
        assignment = dict(zip(game.roles, pair))
        header = f"Game {gnum}/{games}" if games > 1 else ""
        outcome = play_game(client, game, assignment, opts, header=header)
        results.append(outcome)
    if games > 1:
        game.standings([comp_a.label, comp_b.label], results, title="Match result")
    return results


# ──────────────────────────────────────────────────────────────────────────
# Resumable round-robin tournament
# ──────────────────────────────────────────────────────────────────────────

def build_schedule(labels, rounds):
    """Round-robin: every unordered pair plays `rounds` games, swapping the
    first role each round."""
    sched, gid = [], 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            for r in range(rounds):
                a, b = labels[i], labels[j]
                pair = [a, b] if r % 2 == 0 else [b, a]
                sched.append({"id": gid, "players": pair, "status": "pending",
                              "result": None})
                gid += 1
    return sched


def tournament_name(game, labels, rounds):
    h = hashlib.sha1(("|".join(sorted(labels)) + f"#{rounds}#{game}").encode()
                     ).hexdigest()[:8]
    return f"{game}-rr-{h}"


def run_tournament(client, game, competitors, name, rounds, tdir, opts):
    """Round-robin over a dict of label->Competitor. State is flushed atomically
    after every game; re-running the same command resumes at the first pending
    game. Per-game records go to runs/ as usual and are referenced by path."""
    from core.competitor import competitor_from_saved

    os.makedirs(tdir, exist_ok=True)
    state_path = os.path.join(tdir, "tournament.json")
    labels = list(competitors)

    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            tstate = json.load(f)
        saved = tstate.get("competitors")
        if saved:  # saved configs are authoritative on resume
            competitors = {l: competitor_from_saved(l, cfg) for l, cfg in saved.items()}
        labels = tstate["labels"]
        rounds = tstate["rounds"]
        done = sum(1 for g in tstate["schedule"] if g["status"] == "done")
        print(f"{GREEN}Resuming tournament '{name}': "
              f"{done}/{len(tstate['schedule'])} games already played.{RESET}")
    else:
        tstate = {"name": name, "game": game.name, "labels": labels, "rounds": rounds,
                  "competitors": {l: c.to_cfg() for l, c in competitors.items()},
                  "schedule": build_schedule(labels, rounds)}
        atomic_write_json(state_path, tstate)
        print(f"{GREEN}New tournament '{name}': {len(tstate['schedule'])} games "
              f"across {len(labels)} competitors.{RESET}")

    def completed():
        return [g["result"] for g in tstate["schedule"] if g["status"] == "done"]

    while True:
        g = next((g for g in tstate["schedule"] if g["status"] == "pending"), None)
        if g is None:
            break
        idx, total = g["id"] + 1, len(tstate["schedule"])
        assignment = dict(zip(game.roles, (competitors[l] for l in g["players"])))
        outcome = play_game(client, game, assignment, opts,
                            header=f"Game {idx}/{total}")
        keep = {k: outcome.get(k) for k in
                ("labels", "winner", "summary", "forfeit", "turns", "scores",
                 "extra", "stats", "run_dir", "when")}
        g.update({"status": "done", "result": keep})
        atomic_write_json(state_path, tstate)   # crash-safe: flush after each game
        game.standings(labels, completed(),
                       title=f"Standings after {len(completed())} games")

    print(f"\n{BOLD}{GREEN}Tournament '{name}' complete.{RESET}")
    game.standings(labels, completed(), title="FINAL STANDINGS")
    print(f"{DIM}State in {tdir}; game records in {opts['runs_dir']}{RESET}")
