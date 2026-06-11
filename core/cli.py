"""Shared CLI: one entry point (`py play.py <game> [flags]`). Shared flags work
for every game; player-selection flags are generated from each game's declared
role names (--white/--black for chess, etc.); games add their own extras."""

import argparse
import os
import sys

from core import engine
from core.competitor import Competitor, load_roster
from core.ollama import OllamaClient, detect_ollama_url, resolve_model
from core.term import DIM, RESET, utf8_stdout

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = "gemma4:26b"


def get_game(name):
    if name in ("chess",):
        from games.chess import ChessGame
        return ChessGame
    if name in ("ipd", "dilemma"):
        from games.ipd import IPDGame
        return IPDGame
    if name in ("20q", "twentyq", "twenty-questions"):
        from games.twenty_questions import TwentyQuestionsGame
        return TwentyQuestionsGame
    if name in ("codenames", "codebreakers"):
        from games.codenames import CodenamesGame
        return CodenamesGame
    if name in ("go",):
        from games.go import GoGame
        return GoGame
    if name in ("telephone",):
        from games.telephone import TelephoneGame
        return TelephoneGame
    if name in ("werewolf", "mafia"):
        from games.werewolf import WerewolfGame
        return WerewolfGame
    if name in ("scrabble", "scrab"):
        from games.scrabble import ScrabbleGame
        return ScrabbleGame
    return None


GAME_NAMES = ["chess", "ipd", "20q", "codenames", "go", "telephone", "werewolf",
              "scrabble"]


def build_parser(game_cls):
    p = argparse.ArgumentParser(
        prog=f"play.py {game_cls.name}",
        description=f"AI Games — {game_cls.name}: local Ollama models play in the terminal.")
    p.add_argument("--model", default=None,
                   help=f"Competitor for all roles (default: {DEFAULT_MODEL})")
    for role in game_cls.roles:
        p.add_argument(f"--{role}", default=None, metavar="NAME",
                       help=f"Competitor (model or roster label) for {role}")
    p.add_argument("--games", type=int, default=1,
                   help="Head-to-head match length, roles swap each game (default 1)")
    p.add_argument("--tournament", nargs="?", const="", default=None, metavar="NAME",
                   help="Round-robin across a competitor set; resumable by name")
    p.add_argument("--models", default=None,
                   help="Comma-separated models/roster-labels for --tournament "
                        "(default: all installed local, or the whole roster)")
    p.add_argument("--roster", default=None, metavar="PATH",
                   help="YAML roster of named competitors (per-model think + temperature)")
    p.add_argument("--rounds", type=int, default=2,
                   help="Games per pairing in a tournament (default 2)")
    p.add_argument("--include-cloud", action="store_true",
                   help="Include ':cloud' models in --tournament")
    p.add_argument("--no-think", action="store_true",
                   help="Disable model thinking (on by default)")
    p.add_argument("--think-effort", choices=["low", "medium", "high"], default=None,
                   help="Reasoning effort for models that support it (e.g. gpt-oss)")
    p.add_argument("--hide-think", action="store_true",
                   help="Don't stream thinking to the terminal")
    p.add_argument("--num-predict", type=int, default=None,
                   help="Max tokens per reply, shared by thinking + answer "
                        f"(default {getattr(game_cls, 'num_predict_default', 2048)} "
                        "for this game)")
    p.add_argument("--num-ctx", type=int, default=None,
                   help="Ollama context window in tokens — must hold the whole "
                        "prompt PLUS the reply, or generation stops with no "
                        f"answer (default {getattr(game_cls, 'num_ctx_default', 16384)})")
    p.add_argument("--no-comment", dest="no_comment", action="store_true",
                   help="Drop the private-comment field entirely: models aren't asked "
                        "for one and humans aren't prompted")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature (default 0.7)")
    p.add_argument("--retries", type=int, default=2,
                   help="Illegal-action retries before forfeit (default 2)")
    p.add_argument("--max-rounds", type=int, default=None,
                   help=f"Cap on actions per game (default {game_cls.max_rounds_default})")
    p.add_argument("--move-time", type=float, default=None, metavar="SECONDS",
                   help="Flat per-move time limit; failing to act in time loses")
    p.add_argument("--delay", type=float, default=0.0,
                   help="Seconds to pause between moves (default 0)")
    p.add_argument("--runs-dir", default=os.path.join(REPO_DIR, "runs"),
                   help="Where per-game records are written (default ./runs)")
    p.add_argument("--url", default=None,
                   help="Ollama base URL (auto-detects WSL2 host)")
    game_cls.add_args(p)
    return p


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: play.py <game> [flags]\n\ngames: " + ", ".join(GAME_NAMES) +
              "\n\nrun 'play.py <game> --help' for that game's flags")
        return
    game_cls = get_game(argv[0])
    if game_cls is None:
        print(f"Unknown game '{argv[0]}'. Available: {', '.join(GAME_NAMES)}")
        sys.exit(1)

    args = build_parser(game_cls).parse_args(argv[1:])
    utf8_stdout()

    game = game_cls()
    client = OllamaClient(args.url or detect_ollama_url())
    installed = client.list_installed()
    game.configure(args)

    gthink = not args.no_think
    roster = load_roster(args.roster, installed) if args.roster else {}

    def make_adhoc(name, explicit=True):
        if name.lower() == "human":
            return Competitor("human", "human", think=False)
        model = resolve_model(name, explicit=explicit, installed=installed)
        return Competitor(model, model, gthink, args.think_effort, args.temperature)

    def resolve_competitor(name):
        if name is None:
            return None
        return roster[name] if name in roster else make_adhoc(name)

    opts = {
        "retries": args.retries,
        "num_predict": args.num_predict or getattr(game_cls,
                                                   "num_predict_default", 2048),
        "num_ctx": args.num_ctx or getattr(game_cls, "num_ctx_default", 16384),
        "move_time": args.move_time, "delay": args.delay,
        "show_think": not args.hide_think, "no_comment": args.no_comment,
        "max_rounds": args.max_rounds or game_cls.max_rounds_default,
        "runs_dir": args.runs_dir, "record": True,
    }

    print(f"\n{DIM}{'=' * 60}")
    print(f"  Game: {game.name}    Ollama: {client.base_url}")
    clock = f"{args.move_time}s/move" if args.move_time else "none"
    print(f"  Retries: {args.retries}    Max rounds: {opts['max_rounds']}    "
          f"Num-predict: {opts['num_predict']}    Context: {opts['num_ctx']}    "
          f"Clock: {clock}")
    if roster:
        print(f"  Roster: {args.roster}  ({len(roster)} competitors)")
    else:
        think_desc = "off" if not gthink else (args.think_effort or "on")
        print(f"  Off-roster defaults: think={think_desc}  temp={args.temperature}")
    print(f"{'=' * 60}{RESET}")

    def build_field():
        """Competitor set from --models / the roster / all installed local."""
        if roster:
            if args.models:
                sel = [s.strip() for s in args.models.split(",") if s.strip()]
                missing = [s for s in sel if s not in roster]
                if missing:
                    print(f"Roster labels not found: {', '.join(missing)}. "
                          f"Available: {', '.join(roster)}")
                    sys.exit(1)
                return {l: roster[l] for l in sel}
            return dict(roster)
        if args.models:
            comps = [make_adhoc(m.strip())
                     for m in args.models.split(",") if m.strip()]
            field, seen = {}, {}
            for c in comps:
                seen[c.label] = seen.get(c.label, 0) + 1
                if seen[c.label] > 1:   # duplicates get distinct labels
                    c = Competitor(f"{c.label}#{seen[c.label]}", c.model,
                                   c.think, c.effort, c.temperature)
                field[c.label] = c
            return field
        names = [n for n, _ in installed]
        if not args.include_cloud:
            names = [m for m in names if not m.endswith(":cloud")]
        names = list(dict.fromkeys(names))
        return {m: Competitor(m, m, gthink, args.think_effort, args.temperature)
                for m in names}

    try:
        if not game_cls.roles:
            # Chain game (e.g. telephone): the whole field plays in one game.
            if args.tournament is not None:
                print(f"{game.name} runs the whole field in one chain — use "
                      "--games N for repeated chains, not --tournament.")
                sys.exit(1)
            competitors = build_field()
            select = getattr(game, "select_players", None)
            if select is None:
                if len(competitors) < 2:
                    print("Need at least 2 competitors for a chain game.")
                    sys.exit(1)
                print(f"  Chain ({len(competitors)}): "
                      f"{' → '.join(competitors)}")
            elif not competitors:
                print("The model pool is empty.")
                sys.exit(1)
            results = []
            for gnum in range(1, args.games + 1):
                # Pool games (e.g. werewolf) deal a fresh cast every game.
                players = select(competitors) if select else competitors
                game.set_chain(list(players))
                header = f"Game {gnum}/{args.games}" if args.games > 1 else ""
                results.append(engine.play_game(client, game, players,
                                                opts, header=header))
            if args.games > 1:
                game.standings(list(competitors), results,
                               title="Overall standings")
            return
        if args.tournament is not None:
            competitors = build_field()
            if len(competitors) < 2:
                print("Need at least 2 competitors for a tournament.")
                sys.exit(1)
            name = args.tournament or engine.tournament_name(
                game.name, list(competitors), args.rounds)
            tdir = os.path.join(REPO_DIR, "tournaments", name)
            print(f"  Tournament: {name}  ({len(competitors)} competitors, "
                  f"{args.rounds} rounds/pair)")
            engine.run_tournament(client, game, competitors, name, args.rounds,
                                  tdir, opts)
        else:
            if args.model and args.model in roster:
                base = roster[args.model]
            else:
                base = make_adhoc(args.model or DEFAULT_MODEL,
                                  explicit=args.model is not None)
            picked = [resolve_competitor(getattr(args, role)) or base
                      for role in game_cls.roles]
            if len(picked) == 2:
                engine.run_match(client, game, picked[0], picked[1], args.games, opts)
            else:
                assignment = dict(zip(game_cls.roles, picked))
                engine.play_game(client, game, assignment, opts)
    except KeyboardInterrupt:
        print(f"\n{DIM}[Interrupted — completed games are recorded.]{RESET}")
    finally:
        game.close()
