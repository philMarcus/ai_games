#!/usr/bin/env python3
"""Engine + chess tests against the MockClient. Plain asserts, no pytest.
Run from the repo root:  py tests/run_tests.py"""

import io
import json
import os
import re
import shutil
import sys
import tempfile
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess as pychess

from core import engine
from core.competitor import Competitor, competitor_from_saved, load_roster
from games.chess import ChessGame, board_letters, parse_move, san_history
from tests.mockclient import MockClient

PASS = 0


def check(name, fn):
    global PASS
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            fn()
    except Exception as e:
        print(f"FAIL  {name}: {type(e).__name__}: {e}")
        print("      --- captured output ---")
        for line in buf.getvalue().splitlines()[-12:]:
            print(f"      {line}")
        raise
    PASS += 1
    print(f"ok    {name}")


def opts(tmp, **kw):
    o = {"retries": 2, "num_predict": 2048, "move_time": None, "delay": 0,
         "show_think": False, "max_rounds": 200, "runs_dir": tmp, "record": True}
    o.update(kw)
    return o


def mk(label="m1"):
    return Competitor(label, "mock:latest", think=False)


def move_json(m, c="test comment"):
    return json.dumps({"move": m, "comment": c})


# ── chess unit tests ──────────────────────────────────────────────────────

def test_parse_move():
    b = pychess.Board()
    assert parse_move(b, "e4") == pychess.Move.from_uci("e2e4")
    assert parse_move(b, "1. Nf3") == pychess.Move.from_uci("g1f3")
    assert parse_move(b, "e2e4") == pychess.Move.from_uci("e2e4")
    assert parse_move(b, "O-O") is None
    assert parse_move(b, "garbage") is None
    b2 = pychess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1")
    assert parse_move(b2, "0-0") == pychess.Move.from_uci("e1g1")


def test_board_letters_and_history():
    b = pychess.Board()
    for mv in ["e4", "c5", "Nf3"]:
        b.push_san(mv)
    grid = board_letters(b)
    assert grid.splitlines()[0].strip() == "a b c d e f g h"
    assert "5 . . p . . . . . 5" in grid
    assert "4 . . . . P . . . 4" in grid
    assert san_history(b) == "1.e4 c5 2.Nf3"


# ── engine: retries, forfeits, clock ──────────────────────────────────────

def test_retry_then_success():
    game = ChessGame()
    state = game.initial_state()
    client = MockClient(script=[
        "not json at all",
        move_json("Zz9"),           # parseable JSON, illegal move
        move_json("e4"),
    ])
    events = []
    tmp = tempfile.mkdtemp()
    got = engine.take_turn(client, game, state, "white", mk(), opts(tmp), events)
    shutil.rmtree(tmp)
    assert got[0] == "ok", got
    assert state["board"].fen().split()[0].count("P") == 8
    kinds = [e["type"] for e in events]
    assert kinds == ["bad_json", "illegal", "action"], kinds
    # already-tried feedback reached the model on the 3rd call
    last_user = client.calls[2]["messages"][1]["content"]
    assert "already tried" in last_user and '"Zz9"' in last_user


def test_forfeit_after_retries():
    game = ChessGame()
    state = game.initial_state()
    client = MockClient(script=[move_json("Qh9"), move_json("Qh9"), move_json("Qh9")])
    events = []
    tmp = tempfile.mkdtemp()
    got = engine.take_turn(client, game, state, "white", mk(), opts(tmp), events)
    shutil.rmtree(tmp)
    assert got[:2] == ("forfeit", "illegal"), got
    assert len([e for e in events if e["type"] == "illegal"]) == 3


def test_time_forfeit():
    game = ChessGame()
    state = game.initial_state()
    client = MockClient(script=[{"content": "", "timeout": True}])
    events = []
    tmp = tempfile.mkdtemp()
    got = engine.take_turn(client, game, state, "white", mk(),
                           opts(tmp, move_time=60), events)
    shutil.rmtree(tmp)
    assert got[:2] == ("forfeit", "time"), got


# ── engine: a full game with records ──────────────────────────────────────

def smart_responder(model, messages):
    """Plays the first legal move parsed from the FEN in the observation."""
    user = messages[1]["content"]
    fen = re.search(r"FEN:\n(.+)", user).group(1).strip()
    board = pychess.Board(fen)
    mv = next(iter(board.legal_moves))
    return move_json(board.san(mv), "mock move")


def test_full_game_record():
    game = ChessGame()
    client = MockClient(responder=smart_responder)
    tmp = tempfile.mkdtemp()
    o = opts(tmp, max_rounds=6)
    assignment = {"white": mk("alpha"), "black": mk("beta")}
    outcome = engine.play_game(client, game, assignment, o)
    assert outcome["turns"] == 6
    assert outcome["summary"] == "adjudicated at ply cap"
    run_dir = outcome["run_dir"]
    rec_path = os.path.join(run_dir, "record.json")
    assert os.path.exists(rec_path)
    assert os.path.exists(os.path.join(run_dir, "game.pgn"))
    rec = json.load(open(rec_path, encoding="utf-8"))
    assert rec["labels"] == {"white": "alpha", "black": "beta"}
    acts = [e for e in rec["events"] if e["type"] == "action"]
    assert len(acts) == 6 and acts[0]["action"]["comment"] == "mock move"
    assert rec["stats"]["alpha"]["actions"] == 3
    shutil.rmtree(tmp)


def test_forfeit_game_outcome():
    game = ChessGame()
    # white plays e4 fine, black then emits garbage until forfeit
    seq = [move_json("e4"), "x", "x", "x"]
    client = MockClient(script=list(seq))
    tmp = tempfile.mkdtemp()
    outcome = engine.play_game(client, game, {"white": mk("a"), "black": mk("b")},
                               opts(tmp))
    assert outcome["winner"] == "a"
    assert outcome["forfeit"]["label"] == "b"
    assert outcome["forfeit"]["kind"] == "illegal"
    assert outcome["result"] == "1-0"
    shutil.rmtree(tmp)


# ── tournament: schedule, standings, resume ───────────────────────────────

def test_tournament_and_resume():
    tmp = tempfile.mkdtemp()
    tdir = os.path.join(tmp, "t")
    comps = {"alpha": mk("alpha"), "beta": mk("beta")}
    game = ChessGame()
    o = opts(tmp, max_rounds=4)

    engine.run_tournament(MockClient(responder=smart_responder), game, comps,
                          "ttest", 2, tdir, o)
    tstate = json.load(open(os.path.join(tdir, "tournament.json"), encoding="utf-8"))
    assert len(tstate["schedule"]) == 2
    assert all(g["status"] == "done" for g in tstate["schedule"])
    # colors swapped between the two games
    assert tstate["schedule"][0]["players"] == ["alpha", "beta"]
    assert tstate["schedule"][1]["players"] == ["beta", "alpha"]

    # resume: a client that explodes if asked anything proves nothing replays
    def boom(model, messages):
        raise AssertionError("resume should not play any games")
    engine.run_tournament(MockClient(responder=boom), game, comps,
                          "ttest", 2, tdir, o)
    shutil.rmtree(tmp)


# ── competitor / roster ───────────────────────────────────────────────────

def test_competitor_roundtrip():
    c = Competitor("gpt-low", "gpt-oss:20b", think=True, effort="low",
                   temperature=0.6)
    c2 = competitor_from_saved("gpt-low", c.to_cfg())
    assert (c2.model, c2.think, c2.effort, c2.temperature) == \
           (c.model, c.think, c.effort, c.temperature)


def test_roster_load():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "r.yaml")
    with open(path, "w", encoding="utf-8") as f:
        f.write("competitors:\n  fast:\n    model: mock:latest\n    think: off\n"
                "    temperature: 0.3\n")
    roster = load_roster(path, [("mock:latest", 1)])
    assert roster["fast"].think is False and roster["fast"].temperature == 0.3
    shutil.rmtree(tmp)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        check(name, fn)
    print(f"\n{PASS}/{len(tests)} tests passed")
