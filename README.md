# AI Games

A local-model evaluation suite: two [Ollama](https://ollama.ai) models compete or
cooperate across diverse games in your terminal — each game isolating a different
capability — with honest, objective scoring. The opposite of vendor bar charts: you
*watch* models play and *keep legitimate score*.

One shared harness (Ollama client, per-competitor thinking/temperature via YAML
rosters, structured-output moves with private commentary, illegal-action retries →
forfeit, a per-move clock, resumable round-robin tournaments, records-by-default)
plus a small Game interface that each game plugs into. See `PLAN.md` for the design.

**Games:** chess (ported from [ai_chess](https://github.com/philMarcus/ai_chess)).
Planned: Go 9×9, Codenames, iterated prisoner's dilemma, 20 Questions, who's-on-first.

## Requirements

- Ollama running, with at least two models pulled.
- Python 3.9+

```bash
pip install -r requirements.txt    # or: py -m pip install -r requirements.txt
```

## Run

```bash
# one chess game between two models
py play.py chess --white qwen3:14b --black gemma4:26b --no-think

# a 4-game match, roles swap each game
py play.py chess --white gemma4:26b --black gpt-oss:20b --games 4 --move-time 60

# round-robin tournament (resumable: Ctrl+C, rerun the same command to continue)
py play.py chess --tournament bench --models qwen3:14b,phi4,gemma3:12b --rounds 2

# per-competitor settings via a roster
py play.py chess --roster rosters/example.yaml --white gemma-fast --black gemma-slow
```

`py play.py --help` lists games; `py play.py <game> --help` lists that game's flags.

## Rosters

A competitor is a `(model, thinking, temperature)` bundle, not just a model — under a
clock, thinking-off gemma and thinking-on gemma are genuinely different players. Define
named competitors in YAML (see `rosters/example.yaml`); `think` is
`off | on | low | medium | high` (the levels set reasoning effort for models like
gpt-oss). Each label gets its own standings row.

## Records by default

Every game — single, match, or tournament — writes a self-contained record under
`runs/<timestamp>_<game>_<players>/`:

- `record.json` — config, competitors, every action **with its private comment**,
  illegal attempts and feedback, timings, result, and per-player reliability stats
  (actions, illegal/bad-JSON counts, turn of first illegal action, mean move time);
- the game-native transcript alongside it (`game.pgn` for chess — opens in lichess
  with the commentary inline).

Tournament state lives in `tournaments/<name>/tournament.json` and references these
records. Nothing is ephemeral.

## Shared flags (every game)

| Flag | Meaning |
|------|---------|
| `--model NAME`        | Competitor for all roles (default `gemma4:26b`, falls back to largest installed). |
| `--<role> NAME`       | Per-role competitor — model tag or roster label (chess: `--white`/`--black`). |
| `--games N`           | Match length, roles swap each game (default 1). |
| `--tournament [NAME]` | Round-robin across a competitor set; resumable by name. |
| `--models a,b,c`      | Competitors for the tournament (default: all installed local, or the whole roster). |
| `--roster PATH`       | YAML roster of named competitors. |
| `--rounds N`          | Games per pairing in a tournament (default 2). |
| `--no-think` / `--think-effort low\|medium\|high` | Thinking off, or effort level (off-roster default). |
| `--temperature F`     | Sampling temperature (off-roster default, 0.7). |
| `--retries N`         | Illegal-action retries before forfeit (default 2). |
| `--max-rounds N`      | Cap on actions per game (chess default 200 plies; capped games adjudicated). |
| `--move-time SECONDS` | Flat per-move clock; failing to act in time loses ("flag fall"). |
| `--num-predict N`     | Max tokens per reply, shared by thinking + answer (default 2048). |
| `--hide-think`        | Don't stream thinking to the terminal. |
| `--delay SECONDS`     | Pause between moves for watchability. |
| `--url URL`           | Ollama base URL (auto-detects WSL2 host). |

Chess extras: `--board-input` (adds a redundant ASCII letter-grid board to the prompt),
`--eval` / `--engine PATH` (Stockfish average-centipawn-loss scoring).

## Tests

The engine and games are tested against a mock Ollama client — no models needed:

```bash
py tests/run_tests.py
```
