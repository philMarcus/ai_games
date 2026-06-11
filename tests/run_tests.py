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

from core.term import utf8_stdout
utf8_stdout()

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
    assert parse_move(b, "e4") == (pychess.Move.from_uci("e2e4"), None)
    assert parse_move(b, "1. Nf3") == (pychess.Move.from_uci("g1f3"), None)
    assert parse_move(b, "e2e4") == (pychess.Move.from_uci("e2e4"), None)
    assert parse_move(b, "O-O") == (None, None)
    assert parse_move(b, "garbage") == (None, None)
    b2 = pychess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1")
    assert parse_move(b2, "0-0") == (pychess.Move.from_uci("e1g1"), None)


def test_ambiguous_san():
    # knights on a1 and e1 can both reach c2: "Nc2" is ambiguous
    b = pychess.Board("k7/8/8/8/8/8/8/N3N2K w - - 0 1")
    mv, ambiguous = parse_move(b, "Nc2")
    assert mv is None and ambiguous == ["Nac2", "Nec2"], (mv, ambiguous)
    # the disambiguated form parses fine
    assert parse_move(b, "Nac2")[0] == pychess.Move.from_uci("a1c2")
    # through the engine: ambiguous → feedback with candidates → success
    game = ChessGame()
    state = game.initial_state()
    state["board"] = b
    client = MockClient(script=[move_json("Nc2"), move_json("Nac2")])
    events = []
    tmp = tempfile.mkdtemp()
    got = engine.take_turn(client, game, state, "white", mk(), opts(tmp), events)
    shutil.rmtree(tmp)
    assert got[0] == "ok", got
    retry_user = client.calls[1]["messages"][1]["content"]
    assert "AMBIGUOUS" in retry_user and "knights" in retry_user and "c2" in retry_user
    # candidates are deliberately withheld — the model must disambiguate itself
    assert "Nac2" not in retry_user and "Nec2" not in retry_user
    assert "FEEDBACK ON YOUR PREVIOUS REPLY" in retry_user


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


# ── IPD ───────────────────────────────────────────────────────────────────

def ipd_game(chat=0, rounds=2):
    from games.ipd import IPDGame
    g = IPDGame()
    g.chat_rounds = chat
    g.total_rounds = rounds
    return g


def test_ipd_pure():
    g = ipd_game(rounds=2)
    seq = [json.dumps({"action": a, "comment": "c"})
           for a in ("cooperate", "defect", "cooperate", "cooperate")]
    client = MockClient(script=seq)
    tmp = tempfile.mkdtemp()
    outcome = engine.play_game(client, g, {"p1": mk("a"), "p2": mk("b")}, opts(tmp))
    # R1: C vs D -> 0/5; R2: C vs C -> 3/3  => a=3, b=8
    assert outcome["scores"] == {"p1": 3, "p2": 8}, outcome["scores"]
    assert outcome["winner"] == "b"
    assert outcome["cooperation_rate"] == {"p1": 1.0, "p2": 0.5}
    assert os.path.exists(os.path.join(outcome["run_dir"], "match.txt"))
    # observations never reveal the opponent's pending same-round choice
    p2_first_obs = client.calls[1]["messages"][1]["content"]
    assert "cooperate" not in p2_first_obs.split("History:")[1].split("Commit")[0]
    shutil.rmtree(tmp)


def test_ipd_chat():
    g = ipd_game(chat=1, rounds=1)
    seq = [json.dumps({"message": "let us both cooperate, friend", "comment": "c"}),
           json.dumps({"message": "agreed, I will cooperate", "comment": "c"}),
           json.dumps({"action": "defect", "comment": "betrayal"}),
           json.dumps({"action": "cooperate", "comment": "honest"})]
    client = MockClient(script=seq)
    tmp = tempfile.mkdtemp()
    outcome = engine.play_game(client, g, {"p1": mk("a"), "p2": mk("b")}, opts(tmp))
    assert outcome["scores"] == {"p1": 5, "p2": 0}
    # p2 saw p1's message before deciding
    p2_chat_obs = client.calls[1]["messages"][1]["content"]
    assert "let us both cooperate" in p2_chat_obs
    # the betrayal is in the readable export
    txt = open(os.path.join(outcome["run_dir"], "match.txt"), encoding="utf-8").read()
    assert "agreed, I will cooperate" in txt and "p1 DEFECT" in txt
    shutil.rmtree(tmp)


def test_ipd_multi_chat():
    g = ipd_game(chat=2, rounds=1)
    msgs = [f"m{i}" for i in range(1, 5)]
    seq = ([json.dumps({"message": m, "comment": "c"}) for m in msgs]
           + [json.dumps({"action": "cooperate", "comment": "c"}),
              json.dumps({"action": "cooperate", "comment": "c"})])
    client = MockClient(script=seq)
    tmp = tempfile.mkdtemp()
    outcome = engine.play_game(client, g, {"p1": mk("a"), "p2": mk("b")}, opts(tmp))
    assert outcome["scores"] == {"p1": 3, "p2": 3}
    # alternating speakers: p1, p2, p1, p2 — and the decider saw all 4 messages
    decide_obs = client.calls[4]["messages"][1]["content"]
    assert all(m in decide_obs for m in msgs)
    # second exchange happened before any decision
    assert "DECISION step" in decide_obs and "m4" in decide_obs
    shutil.rmtree(tmp)


# ── 20 Questions ──────────────────────────────────────────────────────────

def tq_game(tmp, limit=20):
    import random
    from games.twenty_questions import TwentyQuestionsGame
    g = TwentyQuestionsGame()
    g.limit = limit
    g.rng = random.Random(7)
    g.recent_path = os.path.join(tmp, "recent.json")
    return g


CANDS = ["penguin", "harmonica", "volcano", "submarine", "espresso",
         "giraffe", "lighthouse", "tornado"]


def test_twentyq_win_by_guess():
    tmp = tempfile.mkdtemp()
    g = tq_game(tmp)
    state = g.initial_state()
    events = []
    o = opts(tmp)
    # answerer commits from candidates (harness RNG picks the secret)
    client = MockClient(script=[json.dumps({"candidates": CANDS, "comment": "c"})])
    assert engine.take_turn(client, g, state, "answerer", mk("ans"), o, events)[0] == "ok"
    secret = state["secret"]
    assert secret in CANDS
    assert json.load(open(g.recent_path, encoding="utf-8"))["secrets"] == [secret]
    # asker asks; answerer answers; asker guesses correctly
    client = MockClient(script=[
        json.dumps({"type": "question", "text": "Is it alive?", "comment": "c"})])
    assert engine.take_turn(client, g, state, "asker", mk("ask"), o, events)[0] == "ok"
    assert g.current_role(state) == "answerer"
    client = MockClient(script=[json.dumps({"answer": "no", "comment": "c"})])
    assert engine.take_turn(client, g, state, "answerer", mk("ans"), o, events)[0] == "ok"
    client = MockClient(script=[
        json.dumps({"type": "guess", "text": f"a {secret}", "comment": "c"})])
    assert engine.take_turn(client, g, state, "asker", mk("ask"), o, events)[0] == "ok"
    assert g.is_over(state)
    res = g.result(state)
    assert res["winner"] == "asker" and res["questions_used"] == 2
    shutil.rmtree(tmp)


def test_twentyq_exhaustion():
    tmp = tempfile.mkdtemp()
    g = tq_game(tmp, limit=1)
    state = g.initial_state()
    state["secret"] = "penguin"   # skip commit phase
    events, o = [], opts(tmp)
    client = MockClient(script=[
        json.dumps({"type": "guess", "text": "walrus", "comment": "c"})])
    assert engine.take_turn(client, g, state, "asker", mk("ask"), o, events)[0] == "ok"
    assert g.is_over(state)
    assert g.result(state)["winner"] == "answerer"
    shutil.rmtree(tmp)


# ── Codenames ─────────────────────────────────────────────────────────────

def cn_game(turns=9):
    import random
    from games.codenames import CodenamesGame
    g = CodenamesGame()
    g.turn_limit = turns
    g.rng = random.Random(11)
    return g


def cn_words(state, kind):
    return [w for w in state["words"] if state["kinds"][w] == kind]


def test_codenames_board_setup():
    g = cn_game()
    state = g.initial_state()
    assert len(state["words"]) == 25
    assert len(cn_words(state, "target")) == 9
    assert len(cn_words(state, "assassin")) == 1
    assert len(cn_words(state, "neutral")) == 15
    # guesser observation must not leak the colors; spymaster's grouped map must
    obs = g.observation(state, "guesser")
    assert "SECRET MAP" not in obs and "ASSASSIN:" not in obs
    sobs = g.observation(state, "spymaster")
    assert "TARGETS still hidden (9):" in sobs
    assert f"THE ASSASSIN: {cn_words(state, 'assassin')[0]}" in sobs
    # every word appears in the spymaster map
    assert all(w in sobs for w in state["words"])


def test_codenames_clue_legality():
    g = cn_game()
    state = g.initial_state()
    board_word = state["words"][0]
    assert g.apply(state, "spymaster", {"clue": board_word, "count": 2})[0] == "illegal"
    assert g.apply(state, "spymaster", {"clue": "two words", "count": 2})[0] == "illegal"
    assert g.apply(state, "spymaster", {"clue": "zzz", "count": 99})[0] == "illegal"
    verdict, _ = g.apply(state, "spymaster", {"clue": "concept", "count": 2})
    assert verdict == "ok" and state["clue"] == "concept" and state["guesses_left"] == 3


def test_codenames_guess_flow():
    g = cn_game()
    state = g.initial_state()
    targets = cn_words(state, "target")
    neutral = cn_words(state, "neutral")[0]
    assassin = cn_words(state, "assassin")[0]

    g.apply(state, "spymaster", {"clue": "things", "count": 2})
    # STOP before any guess is illegal
    assert g.apply(state, "guesser", {"guess": "STOP"})[0] == "illegal"
    # target keeps the turn alive
    assert g.apply(state, "guesser", {"guess": targets[0].lower()})[0] == "ok"
    assert state["found"] == 1 and g.current_role(state) == "guesser"
    # re-guessing a revealed word is illegal
    assert g.apply(state, "guesser", {"guess": targets[0]})[0] == "illegal"
    # neutral ends the turn → spymaster again
    assert g.apply(state, "guesser", {"guess": neutral})[0] == "ok"
    assert state["turn"] == 2 and g.current_role(state) == "spymaster"
    # STOP banks a turn after one correct guess
    g.apply(state, "spymaster", {"clue": "more", "count": 1})
    g.apply(state, "guesser", {"guess": targets[1]})
    assert g.apply(state, "guesser", {"guess": "stop"})[0] == "ok"
    assert state["turn"] == 3
    # assassin ends everything
    g.apply(state, "spymaster", {"clue": "doom", "count": 1})
    g.apply(state, "guesser", {"guess": assassin})
    assert g.is_over(state) and state["assassin_hit"] == assassin
    res = g.result(state)
    assert res["extra"]["assassin"] and res["winner"] is None
    assert res["extra"]["found"] == 2


def test_codenames_win():
    g = cn_game()
    state = g.initial_state()
    targets = cn_words(state, "target")
    for w in targets[:-1]:                       # reveal 8 of 9 directly
        state["revealed"][w] = "target"
    state["found"] = 8
    g.apply(state, "spymaster", {"clue": "final", "count": 1})
    verdict, disp = g.apply(state, "guesser", {"guess": targets[-1]})
    assert verdict == "ok" and state["won"] and g.is_over(state)
    res = g.result(state)
    assert res["extra"]["cleared"] and res["scores"]["guesser"] == 9.0


# ── Go ────────────────────────────────────────────────────────────────────

def test_go_coords():
    from games.go import coord_to_xy, xy_to_coord
    assert coord_to_xy("A1") == (0, 0)
    assert coord_to_xy("e5") == (4, 4)
    assert coord_to_xy("J9") == (8, 8)     # J is the 9th column (no I)
    assert coord_to_xy("I5") is None
    assert coord_to_xy("Z3") is None and coord_to_xy("E10") is None
    assert xy_to_coord((8, 8)) == "J9"


def test_go_capture_and_suicide():
    from games.go import GoBoard
    b = GoBoard()
    assert b.play("w", (0, 0))[0] == "ok"
    assert b.play("b", (1, 0))[0] == "ok"
    v, caps = b.play("b", (0, 1))           # fills white's last liberty
    assert v == "ok" and caps == 1
    assert (0, 0) not in b.grid and b.captures["b"] == 1
    # suicide: white back into the (0,0) corner now has no liberties
    v, reason = b.play("w", (0, 0))
    assert v == "illegal" and "suicide" in reason
    # occupied
    assert b.play("w", (1, 0))[0] == "illegal"


def test_go_superko():
    from games.go import GoBoard
    b = GoBoard()
    # classic ko shape around (1,1)/(2,1)
    for color, p in [("b", (1, 0)), ("b", (0, 1)), ("b", (1, 2)),
                     ("w", (2, 0)), ("w", (1, 1)), ("w", (3, 1)), ("w", (2, 2))]:
        assert b.play(color, p)[0] == "ok"
    v, caps = b.play("b", (2, 1))           # black captures the ko stone at (1,1)
    assert v == "ok" and caps == 1
    assert b.simple_ko == (1, 1)
    v, reason = b.play("w", (1, 1))         # immediate recapture forbidden
    assert v == "illegal" and "ko" in reason.lower()
    assert b.play("w", (5, 5))[0] == "ok"   # white must play elsewhere


def test_go_scoring():
    from games.go import GoBoard
    b = GoBoard()
    b.play("b", (0, 0))
    assert b.area_score() == (81, 0)        # all empty space reaches only black
    b.play("w", (8, 8))
    assert b.area_score() == (1, 1)         # shared empty region is neutral


def test_go_full_game_and_sgf():
    from games.go import GoGame
    g = GoGame()
    seq = [json.dumps({"move": m, "comment": f"move {m}"})
           for m in ("E5", "D3", "pass", "pass")]
    client = MockClient(script=seq)
    tmp = tempfile.mkdtemp()
    outcome = engine.play_game(client, g, {"black": mk("b-model"),
                                           "white": mk("w-model")}, opts(tmp))
    # board: 1 black stone, 1 white stone, neutral space → white wins on komi
    assert outcome["scores"] == {"black": 1.0, "white": 8.0}
    assert outcome["winner"] == "w-model"
    sgf = open(os.path.join(outcome["run_dir"], "game.sgf"), encoding="utf-8").read()
    assert sgf.startswith("(;GM[1]FF[4]") and "SZ[9]" in sgf
    assert ";B[ee]" in sgf and ";W[dg]" in sgf      # E5 / D3 in SGF coords
    assert ";B[]" in sgf and "C[move E5]" in sgf    # pass + comment
    shutil.rmtree(tmp)


def test_go_capture_annotations():
    from games.go import GoGame
    g = GoGame()
    state = g.initial_state()
    b = state["board"]
    b.play("w", (0, 0))
    b.play("b", (1, 0))
    b.play("b", (0, 1))          # captures white's corner stone
    # history shown to the model marks the capture
    assert "(captured 1)" in g._history_text(b)
    # the captured player is explicitly told on their next turn
    obs = g.observation(state, "white")
    assert "CAPTURED 1 of your stone(s)" in obs
    # but a non-capturing position has no note
    assert "CAPTURED" not in g.observation(g.initial_state(), "black")


def test_go_illegal_feedback():
    from games.go import GoGame
    g = GoGame()
    state = g.initial_state()
    assert g.apply(state, "black", {"move": "I5", "comment": ""})[0] == "illegal"
    state["board"].play("b", (4, 4))
    v, reason = g.apply(state, "white", {"move": "E5", "comment": ""})
    assert v == "illegal" and "occupied" in reason


# ── Telephone ─────────────────────────────────────────────────────────────

def tel_game(steps=3, stop=False):
    import random
    from games.telephone import TelephoneGame
    g = TelephoneGame()
    g.steps = steps
    g.stop_on_mutation = stop
    g.rng = random.Random(3)
    g.set_chain(["alpha", "beta", "gamma"])
    return g


def test_telephone_chain():
    g = tel_game(steps=3)
    seed = "The quick brown fox jumps over the lazy dog."
    mutant = "The quick brown fox jumped over the lazy dog."
    client = MockClient(script=[
        json.dumps({"text": seed}),            # alpha composes
        json.dumps({"text": seed}),            # beta repeats faithfully
        json.dumps({"text": mutant}),          # gamma mutates
        json.dumps({"text": mutant}),          # alpha repeats the MUTATED text
    ])
    tmp = tempfile.mkdtemp()
    comps = {l: mk(l) for l in ("alpha", "beta", "gamma")}
    outcome = engine.play_game(client, g, comps, opts(tmp))
    assert outcome["extra"]["mutations"] == 1
    assert outcome["extra"]["first_mutation_step"] == 2
    assert outcome["extra"]["per_label"]["beta"]["mutations"] == 0
    assert outcome["extra"]["per_label"]["gamma"]["mutations"] == 1
    # alpha's repeat was judged against the MUTATED text (faithful to it)
    assert outcome["extra"]["per_label"]["alpha"]["mutations"] == 0
    assert mutant in client.calls[3]["messages"][1]["content"]
    assert outcome["extra"]["final"] == mutant
    txt = open(os.path.join(outcome["run_dir"], "telephone.txt"),
               encoding="utf-8").read()
    assert "MUTATED" in txt and "unchanged" in txt
    shutil.rmtree(tmp)


def test_telephone_whitespace_not_mutation():
    g = tel_game(steps=1)
    client = MockClient(script=[
        json.dumps({"text": "Hello   world."}),
        json.dumps({"text": " Hello world.  "}),   # only whitespace differs
    ])
    tmp = tempfile.mkdtemp()
    comps = {l: mk(l) for l in ("alpha", "beta", "gamma")}
    outcome = engine.play_game(client, g, comps, opts(tmp))
    assert outcome["extra"]["mutations"] == 0
    shutil.rmtree(tmp)


def test_telephone_stop_on_mutation():
    g = tel_game(steps=50, stop=True)
    client = MockClient(script=[
        json.dumps({"text": "abc def"}),
        json.dumps({"text": "abc def"}),
        json.dumps({"text": "abc deg"}),           # mutation → game ends
    ])
    tmp = tempfile.mkdtemp()
    comps = {l: mk(l) for l in ("alpha", "beta", "gamma")}
    outcome = engine.play_game(client, g, comps, opts(tmp))
    assert outcome["extra"]["repeats"] == 2
    assert "first mutation at step 2" in outcome["summary"]
    shutil.rmtree(tmp)


# ── truncation feedback ───────────────────────────────────────────────────

def test_truncation_feedback():
    game = ChessGame()
    state = game.initial_state()
    cut = '{"move": "e4", "comment": "a manifesto that never en'
    client = MockClient(script=[
        {"content": cut, "done_reason": "length"},     # cut off mid-JSON
        json.dumps({"move": "e4", "comment": "brief"}),
    ])
    events = []
    tmp = tempfile.mkdtemp()
    got = engine.take_turn(client, game, state, "white", mk(), opts(tmp), events)
    shutil.rmtree(tmp)
    assert got[0] == "ok"
    assert events[0]["type"] == "bad_json" and events[0]["kind"] == "truncated"
    retry_user = client.calls[1]["messages"][1]["content"]
    assert "CUT OFF" in retry_user and "briefly" in retry_user
    assert "not the valid JSON" not in retry_user    # no misleading message


# ── human player & --no-comment ───────────────────────────────────────────

def test_no_comment_schema():
    game = ChessGame()
    state = game.initial_state()
    client = MockClient(script=[json.dumps({"move": "e4"})])
    events = []
    tmp = tempfile.mkdtemp()
    got = engine.take_turn(client, game, state, "white", mk(),
                           opts(tmp, no_comment=True), events)
    shutil.rmtree(tmp)
    assert got[0] == "ok" and got[2] == ""        # no comment returned
    sent = client.calls[0]["schema"]
    assert "comment" not in sent["properties"]
    assert "comment" not in sent.get("required", [])


def feed_stdin(text):
    old = sys.stdin
    sys.stdin = io.StringIO(text)
    return old


def test_human_turn():
    game = ChessGame()
    state = game.initial_state()
    from core.competitor import Competitor
    comp = Competitor("human", "human")
    assert comp.is_human
    events = []
    tmp = tempfile.mkdtemp()
    old = feed_stdin("Zz9\n\ne4\ngood move\n")   # illegal, retry, then legal+comment
    try:
        got = engine.take_turn(None, game, state, "white", comp, opts(tmp), events)
    finally:
        sys.stdin = old
    shutil.rmtree(tmp)
    assert got[0] == "ok" and got[2] == "good move"
    assert state["board"].fen().startswith("rnbqkbnr/pppppppp/8/8/4P3")
    assert [e["type"] for e in events] == ["illegal", "action"]


def test_human_no_comment():
    game = ChessGame()
    state = game.initial_state()
    from core.competitor import Competitor
    comp = Competitor("human", "human")
    events = []
    tmp = tempfile.mkdtemp()
    old = feed_stdin("e4\n")                     # only the move is prompted
    try:
        got = engine.take_turn(None, game, state, "white", comp,
                               opts(tmp, no_comment=True), events)
    finally:
        sys.stdin = old
    shutil.rmtree(tmp)
    assert got[0] == "ok" and got[2] == ""
    assert events[-1]["action"] == {"move": "e4"}


def test_human_eof_forfeits():
    game = ChessGame()
    state = game.initial_state()
    from core.competitor import Competitor
    comp = Competitor("human", "human")
    tmp = tempfile.mkdtemp()
    old = feed_stdin("")                         # input ends immediately
    try:
        got = engine.take_turn(None, game, state, "white", comp, opts(tmp), [])
    finally:
        sys.stdin = old
    shutil.rmtree(tmp)
    assert got[:2] == ("forfeit", "illegal")


# ── Werewolf ──────────────────────────────────────────────────────────────

WW_PLAYERS = ["alpha", "beta", "gamma", "delta", "echo"]


def ww_game(talk=1, wolves=1, seed=5):
    import random
    from games.werewolf import WerewolfGame
    g = WerewolfGame()
    g.n_wolves = wolves
    g.talk_rounds = talk
    g.rng = random.Random(seed)
    g.set_chain(WW_PLAYERS)
    return g


def ww_who(state):
    wolf = [p for p, r in state["role_of"].items() if r == "wolf"][0]
    seer = [p for p, r in state["role_of"].items() if r == "seer"][0]
    return wolf, seer


def ww_responder(plan):
    """plan maps kind -> callable(player, state-free prompt text) or value."""
    def respond(model, messages):
        user = messages[1]["content"]
        if "Send a short private message" in user:
            return json.dumps({"message": "let us strike", "notes": "wolf chat"})
        if "Choose tonight's victim" in user:
            return json.dumps({"target": plan["kill"](user), "notes": "kill note"})
        if "secretly inspect" in user:
            return json.dumps({"target": plan["see"](user), "notes": "saw them"})
        if "Speak to the village" in user:
            return json.dumps({"speech": "I am but a humble villager.",
                               "notes": "talked"})
        if "cast your SEALED vote" in user:
            return json.dumps({"target": plan["vote"](user), "notes": "voted"})
        raise AssertionError("unrecognized werewolf prompt")
    return respond


def test_werewolf_setup_and_village_win():
    import random
    g = ww_game()
    probe = g.initial_state()
    wolf, seer = ww_who(probe)
    assert len([p for p, r in probe["role_of"].items() if r == "villager"]) == 3
    g.rng = random.Random(5)   # the real game must deal the same roles as the probe

    def first_living_nonwolf(user):
        # wolf kills the first valid target listed
        line = user.split("Valid targets: ")[1]
        return line.split(".")[0].split(",")[0].strip()

    def vote_wolf(user):
        if "You are a WEREWOLF" in user:        # the wolf can't vote for itself
            return first_living_nonwolf(user)
        return wolf

    client = MockClient(responder=ww_responder(
        {"kill": first_living_nonwolf, "see": first_living_nonwolf,
         "vote": vote_wolf}))
    tmp = tempfile.mkdtemp()
    comps = {l: mk(l) for l in WW_PLAYERS}
    outcome = engine.play_game(client, g, comps, opts(tmp))
    # everyone votes for the wolf on day 1 → village wins
    assert outcome["extra"]["team"] == "village"
    assert outcome["extra"]["won"][wolf] is False
    assert all(outcome["extra"]["won"][p] for p in WW_PLAYERS if p != wolf)
    story = open(os.path.join(outcome["run_dir"], "story.txt"),
                 encoding="utf-8").read()
    assert "lynch" in story and "private notebooks" in story.lower()
    shutil.rmtree(tmp)


def test_werewolf_notebook_accumulates():
    g = ww_game()
    state = g.initial_state()
    wolf, seer = ww_who(state)
    state["_humans"] = []
    # play the wolf's kill (queue starts with it in a 1-wolf game)
    assert g.current_role(state) == wolf and g._kind(state) == "wolf_kill"
    pool = [p for p in state["alive"] if p != wolf]
    g.apply(state, wolf, {"target": pool[0], "notes": "first entry"})
    assert state["notebooks"][wolf] == ["(night 1) first entry"]
    # the seer acts; then it's day — wolf's next observation shows its notebook
    g.apply(state, seer, {"target": wolf, "notes": "checked"})
    obs = g.observation(state, wolf)
    assert "first entry" in obs
    # seer's private log is in its own observation, not the wolf's
    assert "is a WEREWOLF" in g.observation(state, seer)
    assert "is a WEREWOLF" not in obs


def test_werewolf_wolves_win_and_elimination():
    g = ww_game()
    state = g.initial_state()
    state["_humans"] = []
    wolf, seer = ww_who(state)
    # night 1: wolf kills a plain villager; seer inspects the wolf
    plain = [p for p in state["alive"]
             if state["role_of"][p] == "villager"]
    g.apply(state, wolf, {"target": plain[0], "notes": ""})
    g.apply(state, seer, {"target": wolf, "notes": ""})
    # day phase now; eliminate (forfeit) non-wolves until parity
    alive_nonwolves = [p for p in state["alive"] if p != wolf]
    assert g.eliminate(state, alive_nonwolves[0], "illegal") is True
    assert g.eliminate(state, alive_nonwolves[1], "illegal") is True
    # 1 wolf vs 1 villager left → wolves reach parity
    assert state["over"] and state["team"] == "wolves"
    res = g.result(state)
    assert res["extra"]["won"][wolf] is True


def test_werewolf_redaction():
    g = ww_game()
    state = g.initial_state()
    wolf, seer = ww_who(state)
    # spectator mode: render groups roles, night turns aren't quiet
    state["_humans"] = []
    spec = strip_ansi(g.render(state))
    assert "WOLVES:" in spec and wolf in spec.split("SEER:")[0]
    assert g.quiet_turn(state, wolf) is False
    # human present: roles hidden, others' night turns quiet, kill display vague
    state["_humans"] = ["echo"]
    hidden = strip_ansi(g.render(state))
    assert "WOLVES:" not in hidden and "alive:" in hidden
    assert g.quiet_turn(state, wolf) is True
    pool = [p for p in state["alive"] if p != wolf and state["role_of"][p] != "wolf"]
    verdict, display = g.apply(state, wolf, {"target": pool[0], "notes": "x"})
    assert verdict == "ok" and pool[0] not in display
    assert g.comment_of({}) == ""        # notes never leak with a human present


def test_werewolf_generous_targets():
    g = ww_game()
    state = g.initial_state()
    state["_humans"] = []
    wolf, seer = ww_who(state)
    pool = [p for p in state["alive"] if p != wolf]
    # markdown bold, case, stray punctuation all match
    assert g._match_target(state, f"**{pool[0]}**", pool) == pool[0]
    assert g._match_target(state, pool[0].upper() + ".", pool) == pool[0]
    assert g._match_target(state, "nobody-real", pool) is None


def test_werewolf_pool_and_names():
    import random
    from core.competitor import Competitor
    g = ww_game()
    g.rng = random.Random(9)
    g.n_players = 7
    pool = {"gemma4:26b": Competitor("gemma4:26b", "gemma4:26b"),
            "qwen3:14b": Competitor("qwen3:14b", "qwen3:14b"),
            "human": Competitor("human", "human")}
    players = g.select_players(pool)
    assert len(players) == 7
    # the human is always seated, exactly once
    humans = [c for c in players.values() if c.is_human]
    assert len(humans) == 1
    # table names: base stripped at the colon, dash, Codenames word
    for name, c in players.items():
        base = "human" if c.is_human else c.model.split(":")[0]
        assert name.startswith(base + "-") and ":" not in name
    # 3-model pool fills 7 seats → duplicates of underlying models exist
    assert len({c.model for c in players.values()}) <= 3
    # wolves scale with the table size
    from games.werewolf import default_wolves
    assert default_wolves(5) == 1 and default_wolves(7) == 2
    assert default_wolves(10) == 3 and default_wolves(15) == 4


def strip_ansi(s):
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


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
