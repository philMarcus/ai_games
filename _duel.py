#!/usr/bin/env python3
"""Step a chess game between two externally-driven players, using the real
ai_games chess rules (legality, ambiguity, prompts, PGN). Used to relay a
Claude-vs-Claude match: the orchestrator spawns a fresh stateless agent per
move and feeds its JSON here.

  py _duel.py init                      start a new game, print white's prompt
  echo '{"move":"e4","comment":"..."}' | py _duel.py play
                                        apply side-to-move's action; prints
                                        result, board, and the next observation
"""

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.term import utf8_stdout
from games.chess import ChessGame

utf8_stdout()

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "_duel_state.json")
LABELS = {"white": "claude-fable-5", "black": "claude-opus-4.8"}
MAX_PLIES = 80


def load():
    game = ChessGame()
    state = game.initial_state()
    meta = {"moves": [], "comments": []}
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as f:
            meta = json.load(f)
        for u in meta["moves"]:
            state["board"].push_uci(u)
        if meta["moves"]:
            state["last"] = state["board"].peek()
        state["comments"] = list(meta["comments"])
    return game, state, meta


def save(meta):
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)


def show_prompt(game, state, role):
    print(f"=== SYSTEM PROMPT ({role}) ===")
    print(game.system_prompt(role))
    print(f"=== OBSERVATION ({role} to move) ===")
    print(game.observation(state, role))


def finish(game, state, meta, capped):
    res = game.result(state, capped=capped)
    winner_label = LABELS.get(res["winner"]) if res["winner"] else None
    print(f"GAME OVER: {res['summary']}  |  winner: {winner_label or 'draw'}")
    outcome = dict(res)
    outcome.update({"game": "chess", "labels": LABELS, "models": LABELS,
                    "winner": winner_label,
                    "turns": len(meta["moves"]),
                    "when": datetime.now().isoformat(timespec="seconds")})
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(HERE, "runs", f"{ts}_chess_claude-fable-5_vs_claude-opus-4.8")
    os.makedirs(run_dir, exist_ok=True)
    record = dict(outcome)
    record["moves"] = meta["moves"]
    record["comments"] = meta["comments"]
    with open(os.path.join(run_dir, "record.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    game.export(state, outcome, run_dir)
    print(f"recorded: {run_dir}")
    os.remove(STATE)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "obs"

    if cmd == "init":
        if os.path.exists(STATE):
            os.remove(STATE)
        game, state, meta = load()
        save(meta)
        print(game.render(state))
        print()
        show_prompt(game, state, "white")
        return

    game, state, meta = load()
    role = game.current_role(state)

    if cmd == "obs":
        show_prompt(game, state, role)
        return

    if cmd == "play":
        action = json.loads(sys.stdin.read())
        verdict, info = game.apply(state, role, action)
        if verdict == "illegal":
            print(f"ILLEGAL ({role}): {info}")
            sys.exit(2)
        meta["moves"] = [m.uci() for m in state["board"].move_stack]
        meta["comments"].append(str(action.get("comment", "")).strip())
        save(meta)
        print(f"PLAYED ({role}): {info}")
        comment = str(action.get("comment", "")).strip()
        if comment:
            print(f'  "{comment}"')
        print(game.render(state))
        if game.is_over(state):
            finish(game, state, meta, capped=False)
        elif len(meta["moves"]) >= MAX_PLIES:
            finish(game, state, meta, capped=True)
        else:
            print()
            nrole = game.current_role(state)
            print(f"=== OBSERVATION ({nrole} to move) ===")
            print(game.observation(state, nrole))
        return

    print(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
