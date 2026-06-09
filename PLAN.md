# AI Games — a local-model evaluation suite

> **Status: plan only.** One project, one shared harness, many games. Two local
> Ollama models compete or cooperate across diverse games — each isolating a
> different capability — with honest, objective scoring wherever possible. The
> opposite of vendor bar charts: you *watch* models play and *keep legitimate
> score*. Supersedes the standalone `ai_go` plan (folded in below).

Working reference today: `C:\Users\Phil\chess\ai_chess.py` (GitHub
philMarcus/ai_chess) already contains ~90% of the shared harness. Who's-on-first
(`C:\Users\Phil\whos_on_first`) is the dialogue-type prototype. Both migrate in.

## Why one project

Chess, Go, Codenames, prisoner's dilemma, 20 Questions and who's-on-first differ
only in **three knobs**: the **rules/legality**, **what's hidden** from each agent,
and **how you score**. Everything else — talking to Ollama, per-model think/temp
configs, structured-output parsing, retries→forfeit, the per-move clock, resumable
round-robin tournaments, standings, transcript export — is identical. Building each
as its own script triplicates that machinery. A single harness with a small "Game"
interface lets each game plug in and inherit all of it.

## The shared harness (already built in ai_chess — extract to `core/`)

- **Ollama client:** WSL-aware URL detection, model list/resolve, **structured-output
  streaming chat** with live thinking, **reasoning effort** (low/med/high), and the
  **deadline cutoff** that powers the clock.
- **Competitor + YAML roster:** a competitor is `(model, think off/on/low/med/high,
  temperature)` — so thinking-on vs thinking-off of one model are distinct entrants.
  Fully game-agnostic.
- **Agent turn loop:** build prompt → structured JSON action → validate → **retry with
  feedback** (accumulating already-tried illegal actions) → **forfeit** (illegal) /
  **time-forfeit** (clock). Game-agnostic.
- **Per-move clock** (flat `--move-time`), **resumable round-robin tournaments**
  (atomic state, skip-completed resume), **standings**, config banner, UTF-8 stdout.
- Extract from ai_chess **HEAD** so recent behavior (`--board-input` letter grid,
  accumulated illegal-move feedback, comment shown on illegal moves) carries over.
- **Not carried over: incompetence benching.** ai_chess benches a model after 2
  consecutive illegal-move forfeits; in practice essentially *all* current local models
  go illegal in the chess midgame, so the bench just disqualifies the whole field.
  Drop it from the core. (A possible future replacement: an optional per-game
  *competence gate* — e.g. a short qualifier — if models ever get reliable enough for
  it to be meaningful. Not in scope now.)

## The Game interface (the three knobs, formalized)

A game module provides:

- `initial_state()` — board / cards / secret / payoff matrix.
- `observation(state, role)` — the prompt text an agent sees, **respecting hidden info**.
- `action_schema(role)` — JSON schema for that role's move (e.g. `{move, comment}`,
  `{clue, count}`, `{action, message}`, `{question}` / `{answer}`).
- `apply(state, role, action)` — **legality check + transition**; returns
  `ok | illegal(reason) | gameover`. Illegal feeds the existing retry/forfeit path.
- `result(state)` — outcome + score(s) + a per-game **stats dict** (e.g.
  rounds-to-first-illegal, move times) persisted into tournament state.
- `standings(results)` — **game-defined leaderboard aggregation**. The chess-shaped
  W/L/D table is NOT a core invariant: IPD ranks by accumulated payoff points;
  cooperative Codenames reports a cluegiver×guesser matrix, not a ranking. The core
  scheduler/persistence is shared; the table each game prints is its own.
- `render(state)` — human terminal display (e.g. the colored chess board).
- `transcript(state)` — export (PGN / SGF / JSON log) with private comments embedded.
- `meta` — players, **role names** (the shared CLI maps generic player flags onto
  game-declared roles — `--white/--black` is chess vocabulary, not core), hidden-info?,
  cooperative vs competitive, scoring type, **turn structure** (below).

**Turn structures the engine supports** (games aren't all "alternating"):
1. **Alternating, perfect info** — chess, Go, Connect Four.
2. **Role-paired** (two roles, possibly cooperative) — Codenames spymaster↔guesser,
   20Q asker↔answerer.
3. **Simultaneous hidden action** (optionally preceded by a chat phase) — prisoner's
   dilemma.
4. **Free dialogue / performance** (no legality, judge- or exhibition-scored) —
   who's-on-first, debate.

## Cross-cutting eval metrics (standardized by the harness)

Beyond per-game win/loss, the harness reports the same **reliability** axes everywhere:
**rounds-to-first-illegal**, forfeit rate (illegal vs time), mean move time / tokens, and
clock flags. These often separate close models better than wins do (in chess, nearly every
current local model goes illegal in the midgame — *when* is the discriminating number).
These stats are stored per game in tournament state, not just printed. Per-game scores are
**not comparable across games** (a chess result ≠ an IPD score) — each game keeps its own
leaderboard; a normalized cross-game ranking is a possible later add.

## Everything on disk (records by default)

This is a **qualitative study** — the primary artifact is the games themselves, not a
stats table. Today only tournaments persist results (single chess games leave just a
PGN). In ai_games, **every** game — single, match, or tournament — automatically writes
a self-contained record under `runs/`, e.g. `runs/2026-06-09_chess_gemma-fast_vs_gpt-low/`:

- `record.json` — config + competitors, every action **with its private comment**,
  illegal attempts and retry feedback, per-move timings, result, stats dict;
- the game-native export alongside it (PGN / SGF / dialogue log / Q&A log).

Nothing is ephemeral: any game you watched (or ran overnight) can be reread, dropped
into a Claude conversation for post-hoc review, or mined later if a question becomes
interesting. Tournament state lives in its own folder and references these per-game
records rather than duplicating them.

---

## Games

### 1. Chess  *(exists → migrate first)*
- **Tests:** planning, board state-tracking, rule-following. **Turn:** alternating.
- **Action:** `{move (SAN), comment}`. **Legality:** python-chess (parse_san).
- **Score:** win/loss + ply-to-first-illegal + optional **Stockfish ACPL**. **Export:** PGN.

### 2. Who's on First  *(exists → migrate later)*
- **Tests:** improv, persona, comedic timing, theory of mind. **Turn:** free dialogue.
- **Action:** free text (in-character line). **Legality:** none. **Score:** exhibition or
  optional **LLM-judge** (note judge bias). **Export:** dialogue log.

### 3. Go 9×9  *(folded from the old ai_go plan)*
- **Tests:** spatial/holistic reasoning, capture/liberty tracking — much harder for LLMs
  than chess (start at 9×9 to keep games legible and short). **Turn:** alternating.
- **Action:** `{move (GTP coord e.g. E5, or "pass"), comment}`.
- **Legality:** in-house `GoBoard` (place / capture by flood-fill liberties / suicide /
  **positional superko**), or a library (`sente`/`gomill`). Illegal → forfeit.
- **End/score:** two passes → **Tromp–Taylor area scoring** + `--komi` (default 7.0,
  programmatic and unambiguous); optional **KataGo** for eval (ACPL analog) and
  human-correct scoring. **Render:** ● green / ○ white grid, star points, last-move bg.
  **Export:** SGF with comments.
- **Open:** in-house board vs library; Tromp–Taylor vs KataGo scoring; KataGo now or later.

### 4. Codenames
- **Tests:** **theory of mind** (cluegiver modeling what the guesser will infer) +
  semantic precision + risk management. **Turn:** role-paired (spymaster↔guesser).
- **Configs:** (A) **cooperative role-split** — model A clues, model B guesses, vs the
  board; score = turns-to-clear / words-hit, assassin = instant fail. Round-robin
  cross-pairing reveals good *cluegivers* vs good *guessers* (different skills). (B)
  **team-vs-team** — A=red, B=blue on a shared board, first to clear wins (clean head-to-head).
- **Action:** spymaster `{clue, count, reasoning}`; guesser guesses **one word at a
  time** — `{guess, reasoning}` or `{stop}` — seeing the revealed color of each guess
  before deciding to continue (up to count+1) or bank the turn. Committing an ordered
  list up front would delete the risk-management decision the game is famous for.
- **Legality (programmatic):** clue is one word, not on the board, not a derivative/substring.
- **Score:** objective (turns / words / assassin) or win/loss. **Data:** a 25-word board
  drawn from a **generic noun list we source/ship ourselves** (not the copyrighted card
  set); harness holds the hidden color map. **Export:** clue/guess log.

### 5. Iterated Prisoner's Dilemma (Axelrod) — with optional chat
- **Tests:** strategy, cooperation, and (with chat) **trustworthiness & deception**.
  **Turn:** simultaneous hidden action, optional pre-round chat phase.
- **Modes:** **pure** (no talk — the classic Axelrod tournament, isolates strategy like
  tit-for-tat) and **`--chat`** (negotiate in natural language, *then* secretly choose) —
  the LLM-native version that tests whether a model honors its word.
- **Horizon must be hidden or randomized.** With a known round count, backward induction
  makes defect-always the rational line and the match unravels from the last round. Tell
  players the game has "an unknown number of rounds"; internally use a fixed count or a
  per-round continuation probability (Axelrod's standard fix).
- **Chat protocol (chat mode):** a fixed number of alternating messages per round
  (default 1 each), first speaker alternating by round; then both secretly commit.
- **Action:** `{action: cooperate|defect}` (+ `{message}` in chat mode). Both revealed
  simultaneously each round; payoff matrix tallied.
- **Score:** total points (round-robin = Axelrod tournament, drops into standings as-is),
  plus **cooperation rate**, **betrayal-after-promising rate** (a novel trust metric),
  exploitability. **Export:** per-round actions + messages.

### 6. Twenty Questions
- **Tests:** deductive question strategy + honesty. **Turn:** role-paired (asker↔answerer).
- **Mechanic:** answerer **commits its secret up front** (harness-stored, hidden from
  the asker) — an **integrity mechanism**, not a per-answer correctness score: it pins
  the target so the answerer can't drift to a different word mid-game when cornered.
  Whether individual yes/no answers were fair is left to **informal post-hoc review of
  the saved transcript** (you/Claude reading the record against the committed secret),
  in keeping with the qualitative aim — no predetermined word dataset, no scored judge.
- **Secret variety:** a naive "think of a word" prompt mode-collapses — the same model
  picks "elephant" every run. Inject **harness-side entropy**: have the answerer propose
  ~20 candidate secrets and let the harness RNG pick one (the model supplies creativity,
  the dice supply randomness), optionally seeded with a random category / first-letter
  constraint; keep a small history of recent secrets (in the run records) to exclude
  repeats within and across sessions.
- **Action:** asker `{question}` or `{guess}`; answerer `{answer: yes|no|unsure}`.
- **Score:** questions-to-solve. **Export:** Q&A log (with the committed secret).

---

## Proposed repo structure

```
ai_games/
  core/
    ollama.py       # client (injectable), streaming+thinking, clock cutoff, model resolve
    competitor.py   # Competitor + YAML roster
    engine.py       # turn loop, retries/forfeit, tournaments, persistence
    game.py         # the Game interface / base class + turn-structure runners
    transcript.py   # PGN / SGF / JSON log helpers
    cli.py          # shared flags (--roster/--tournament/--move-time…) + per-game
                    # player flags mapped onto each game's declared role names
  games/
    chess.py  go.py  codenames.py  ipd.py  twenty_questions.py  whos_on_first.py
  tests/            # engine/game tests against the mock client
  data/             # codenames word lists, etc.
  rosters/          # YAML competitor rosters
  runs/             # auto-saved per-game records (gitignored)
  README.md  requirements.txt
```
One entry point: `python -m ai_games <game> [shared flags] [game-specific flags]`.
Shared flags (roster, tournament, clock, retries, think) work for every game; each game
adds only its own (e.g. `--komi`, `--chat`, `--assassin-loss`). Player-selection flags
come from the game's role names, not hardcoded `--white/--black`.

## Testing: a mock Ollama client (build it first)

Real-model runs are slow and nondeterministic — verifying the chess harness meant
watching multi-minute games. The Ollama client is **injectable**: a `MockClient` that
returns scripted JSON responses (including bad JSON, illegal actions, slow "thinking,"
and deadline overruns) lets the entire engine — turn loops, retries, forfeits, clock,
tournament resume, game-defined standings — be unit-tested in milliseconds, and each
game's rules/scoring be tested without any model. This is the highest-leverage piece of
the refactor: the core extraction is exactly when regressions sneak in. Real-model runs
stay as the fun part, not the test suite.

## Migrating the existing apps

- **Extract** the reusable core out of `ai_chess.py` into `core/` (ollama, competitor/roster,
  engine, tournament, transcript), then **re-express chess as `games/chess.py`** on the Game
  interface. Keep `ai_chess.py` working as a parity reference until the port matches it.
- **Who's-on-first** becomes `games/whos_on_first.py` (free-dialogue type), reusing the
  Ollama client + roster; scoring is exhibition or an optional LLM-judge.

## Build order

1. **Extract core** from ai_chess HEAD + define the Game interface (no new behavior,
   minus benching) + the **MockClient and first engine tests**.
2. **Port chess** onto it (proves the abstraction; parity-check against ai_chess).
3. **IPD** (+chat) and **20 Questions** — cheap, and they exercise the *simultaneous* and
   *role-paired/hidden-secret* turn structures the interface must support.
4. **Codenames** — richest payoff (word lists, cooperative scoring, cross-pairing).
5. **Go 9×9** — most new game logic (GoBoard, scoring, SGF).
6. **Fold in who's-on-first** (dialogue/performance type + optional judge).

## Open decisions

- **Package vs single dispatcher file** (`python -m ai_games <game>` vs one big script).
- **How much turn-structure generality to build now** vs add patterns per game as needed.
- **Cross-game ranking:** per-game leaderboards only (recommended) vs a later normalized
  meta-ranking. Scores aren't comparable across games.
- **LLM judge** for subjective games (who's-on-first, debate) — accept judge bias, or keep
  those exhibition-only.
- **Keep ai_chess / whos_on_first as standalone repos** too, or fully absorb them.
- Go specifics (carried over): in-house `GoBoard` vs library; Tromp–Taylor vs KataGo scoring;
  set up KataGo now or later.
