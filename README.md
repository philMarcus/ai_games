# AI Games

A local-model evaluation suite: two [Ollama](https://ollama.ai) models compete or
cooperate across diverse games in your terminal — each game isolating a different
capability — with honest, objective scoring. The opposite of vendor bar charts: you
*watch* models play and *keep legitimate score*.

One shared harness (Ollama client, per-competitor thinking/temperature via YAML
rosters, structured-output moves with private commentary, illegal-action retries →
forfeit, a per-move clock, resumable round-robin tournaments, records-by-default)
plus a small Game interface that each game plugs into. See `PLAN.md` for the design.

**Games:**

- **chess** — planning, board state-tracking, rule-following (ported from
  [ai_chess](https://github.com/philMarcus/ai_chess); SAN with disambiguation,
  PGN export, optional Stockfish ACPL).
- **ipd** — iterated prisoner's dilemma, Axelrod-style: standings rank by
  accumulated points. `--chat` adds a pre-round message exchange, so you can
  watch models promise, persuade, and betray. The round count is hidden from
  the players (a known horizon unravels into defect-always).
- **20q** — Twenty Questions: the answerer proposes candidate secrets and the
  harness dice pick one (committed up front so it can't drift; recent secrets
  are excluded for variety), then the asker deduces it in `--questions` tries.
- **codenames** — cooperative: the spymaster sees the hidden 9-target/1-assassin
  map and gives one-word clues; the guesser guesses one word at a time and may
  STOP to bank the turn. Theory of mind + risk management; the assassin ends
  everything. Standings rank by average targets found.
- **go** — 9×9 Go, the spatial-reasoning stress test: in-house rules (captures,
  suicide, positional superko), GTP coordinates (no column I), two passes end
  the game, Tromp–Taylor area scoring + `--komi`, SGF export with commentary.
- **telephone** — the kids' game, played by your whole stable at once: the
  first model composes a `--length phrase|sentence|paragraph`, then every model
  in rotation must repeat the previous output EXACTLY for `--steps` turns
  (default 50). Any change is a mutation that propagates down the chain.
  Standings rank models by transcription fidelity.

Planned: who's-on-first.

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

# prisoner's dilemma with negotiation; 20 Questions; Codenames
py play.py ipd --p1 gemma4:26b --p2 gpt-oss:20b --chat --no-think
py play.py 20q --answerer gemma4:26b --asker qwen3:14b --no-think
py play.py codenames --spymaster gemma4:26b --guesser qwen3:14b --no-think
py play.py go --black gemma4:26b --white gpt-oss:20b --no-think --move-time 60

# telephone through every installed local model (or pick the chain with --models)
py play.py telephone --no-think
py play.py telephone --models gemma4:26b,qwen3:14b,phi4 --length paragraph --steps 30

# play a model yourself: "human" is a valid competitor anywhere a model name goes
py play.py chess --white human --black gemma4:26b --no-comment
py play.py codenames --spymaster gemma4:26b --guesser human
```

## Playing as a human

Use `human` as the competitor name for any role (or in `--models` for telephone).
On your turn the game state is printed and you're prompted **per field** of the
action (e.g. `move>`, then `comment>`); enums and numbers are validated as you
type. Humans get unlimited retries on illegal actions — but `--move-time` still
applies, so you can lose on the clock. With `--no-comment`, the comment field
disappears for everyone: models aren't asked for one, humans aren't prompted.

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
| `--no-comment`        | Drop the private-comment field: models aren't asked, humans aren't prompted. |
| `--hide-think`        | Don't stream thinking to the terminal. |
| `--delay SECONDS`     | Pause between moves for watchability. |
| `--url URL`           | Ollama base URL (auto-detects WSL2 host). |

Chess extras: `--board-input` (adds a redundant ASCII letter-grid board to the prompt),
`--eval` / `--engine PATH` (Stockfish average-centipawn-loss scoring).
IPD extras: `--chat [N]` (N messages each before every decision; bare `--chat` = 1),
`--ipd-rounds N` (hidden from the players).
20q extras: `--questions N` (asker's budget, default 20).
Codenames extras: `--turns N` (clue turns to find all 9 targets, default 9).
Go extras: `--komi F` (White's compensation, default 7.0).
Telephone extras: `--length phrase|sentence|paragraph`, `--steps N` (default 50),
`--stop-on-mutation` (end at the first change instead).

## Tests

The engine and games are tested against a mock Ollama client — no models needed:

```bash
py tests/run_tests.py
```
