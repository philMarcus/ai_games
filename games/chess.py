"""Chess on the shared harness — ported from ai_chess.py. Tests planning, board
state-tracking, and rule-following. Legality, draw/mate detection and PGN come
from python-chess; an illegal move (after retries) forfeits."""

import os
import re
import shutil
from datetime import date

import chess
import chess.pgn

from core.game import Game
from core.term import (BOLD, CYBER, DIM, HILITE_BG, RED, RESET, VS_TEXT, WHITE,
                       YELLOW)

try:
    import chess.engine  # only needed for --eval
except Exception:  # pragma: no cover
    pass

MOVE_SCHEMA = {
    "type": "object",
    "properties": {
        "move": {"type": "string"},
        "comment": {"type": "string"},
    },
    "required": ["move", "comment"],
}

PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}

GLYPHS = {
    chess.PAWN: "♙" + VS_TEXT, chess.KNIGHT: "♘" + VS_TEXT,
    chess.BISHOP: "♗" + VS_TEXT, chess.ROOK: "♖" + VS_TEXT,
    chess.QUEEN: "♕" + VS_TEXT, chess.KING: "♔" + VS_TEXT,
}

SYSTEM_TMPL = (
    "You are a strong chess player playing a real game as {color}.\n"
    "On your turn you are given the current position. Reply with ONLY a JSON "
    'object of the form {{"move": "<SAN>", "comment": "<text>"}}.\n'
    "- \"move\" must be a single legal move in Standard Algebraic Notation "
    "(e.g. e4, Nf3, O-O, exd5, e8=Q, Qxf7#). It must be legal in the position "
    "shown — an illegal or impossible move forfeits the entire game. If two of "
    "your pieces of the same type can reach the square, disambiguate as SAN "
    "requires (e.g. Ngf6, Rad1).\n"
    "- \"comment\" is one or two sentences on your plan. It is PRIVATE: your "
    "opponent never sees it, so be candid.\n"
    "Do not resign and do not offer draws — just play your best legal move.\n"
    "Decide promptly: keep any private reasoning brief and then commit to a move. "
    "Do not deliberate at length — output your JSON answer without endless analysis."
)


# ──────────────────────────────────────────────────────────────────────────
# Board helpers (ported intact from ai_chess)
# ──────────────────────────────────────────────────────────────────────────

def render_board(board, last_move=None):
    """Colorized Unicode board from White's perspective. Pieces always keep
    their own colour; the last move is shown by a grey square background."""
    lines = []
    for rank in range(7, -1, -1):
        cells = []
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            hl = last_move is not None and sq in (last_move.from_square,
                                                  last_move.to_square)
            bg = HILITE_BG if hl else ""
            if piece is None:
                cell = f"{bg}{DIM}·{RESET}"
            else:
                glyph = GLYPHS[piece.piece_type]
                color = f"{BOLD}{WHITE}" if piece.color == chess.WHITE else f"{BOLD}{CYBER}"
                cell = f"{bg}{color}{glyph}{RESET}"
            cells.append(cell)
        lines.append(f"{DIM}{rank + 1}{RESET} " + " ".join(cells))
    lines.append(f"  {DIM}a b c d e f g h{RESET}")
    return "\n".join(lines)


def board_letters(board):
    """ASCII letter grid for the model prompt: uppercase=White, lowercase=Black,
    '.'=empty, with a-h / 1-8 labels. Redundant with the FEN, on purpose."""
    rows = ["  a b c d e f g h"]
    for rank in range(7, -1, -1):
        cells = []
        for f in range(8):
            piece = board.piece_at(chess.square(f, rank))
            cells.append(piece.symbol() if piece else ".")
        rows.append(f"{rank + 1} " + " ".join(cells) + f" {rank + 1}")
    rows.append("  a b c d e f g h")
    return "\n".join(rows)


def san_history(board):
    """Replay move_stack into a numbered SAN movetext string."""
    tmp = chess.Board()
    parts = []
    for i, mv in enumerate(board.move_stack):
        san = tmp.san(mv)
        if i % 2 == 0:
            parts.append(f"{i // 2 + 1}.{san}")
        else:
            parts[-1] += f" {san}"
        tmp.push(mv)
    return " ".join(parts) if parts else "(none yet — this is the opening move)"


def material_diff(board):
    w = sum(PIECE_VALUES[p.piece_type] for p in board.piece_map().values()
            if p.color == chess.WHITE)
    b = sum(PIECE_VALUES[p.piece_type] for p in board.piece_map().values()
            if p.color == chess.BLACK)
    return w - b


def normalize_move_str(s):
    s = s.strip().strip('."` \t').strip()
    s = re.sub(r"^\d+\.+\s*", "", s)          # strip leading "12." / "12..."
    s = s.split()[0] if s.split() else s      # keep only the first token
    s = s.replace("0-0-0", "O-O-O").replace("0-0", "O-O")
    return s


AMBIG_SAN_RE = re.compile(r"^([NBRQK])[a-h]?[1-8]?x?([a-h][1-8])")


def ambiguous_candidates(board, s):
    """Disambiguated SANs of all legal moves matching an ambiguous SAN string
    (same piece type, same destination), e.g. 'Nf6' -> ['Ndf6', 'Ngf6']."""
    m = AMBIG_SAN_RE.match(s)
    if not m:
        return []
    piece = chess.Piece.from_symbol(m.group(1)).piece_type
    to_sq = chess.parse_square(m.group(2))
    return sorted(board.san(mv) for mv in board.legal_moves
                  if mv.to_square == to_sq
                  and board.piece_type_at(mv.from_square) == piece)


def parse_move(board, raw):
    """Parse a model's move string (SAN preferred, UCI fallback). Returns
    (move, None) on success, (None, candidate_sans) when the SAN was legal but
    ambiguous, or (None, None) when illegal/unparseable."""
    s = normalize_move_str(str(raw))
    if not s:
        return None, None
    try:
        return board.parse_san(s), None
    except chess.AmbiguousMoveError:
        return None, ambiguous_candidates(board, s)
    except Exception:
        pass
    try:
        mv = board.parse_uci(s.lower())
        if mv in board.legal_moves:
            return mv, None
    except Exception:
        pass
    return None, None


# ──────────────────────────────────────────────────────────────────────────
# Optional Stockfish evaluation
# ──────────────────────────────────────────────────────────────────────────

class Evaluator:
    """Wraps Stockfish to measure per-move centipawn loss (ACPL)."""

    def __init__(self, path, depth=12):
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.depth = depth

    def _cp(self, board):
        info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
        return info["score"].pov(board.turn).score(mate_score=10000)

    def loss(self, board_before, move):
        try:
            best = self._cp(board_before)
            after = board_before.copy()
            after.push(move)
            got = -self._cp(after)            # flip opponent-POV back to mover
            return max(0, best - got)
        except Exception:
            return None

    def close(self):
        try:
            self.engine.quit()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────
# The game
# ──────────────────────────────────────────────────────────────────────────

class ChessGame(Game):
    name = "chess"
    roles = ("white", "black")
    max_rounds_default = 200    # plies

    def __init__(self):
        self.board_input = False
        self.evaluator = None

    @classmethod
    def add_args(cls, p):
        p.add_argument("--board-input", action="store_true",
                       help="Also give the model a redundant ASCII letter-grid board "
                            "(default: FEN + history only)")
        p.add_argument("--eval", action="store_true",
                       help="Stockfish ACPL scoring (needs a stockfish binary)")
        p.add_argument("--engine", default=None,
                       help="Path to a Stockfish binary (else auto-detected)")

    def configure(self, args):
        self.board_input = args.board_input
        if args.eval:
            path = args.engine or shutil.which("stockfish")
            if not path:
                print(f"{YELLOW}--eval requested but no Stockfish binary found "
                      f"(install one or pass --engine PATH); continuing without "
                      f"eval.{RESET}")
            else:
                try:
                    self.evaluator = Evaluator(path)
                    print(f"{DIM}Stockfish eval enabled: {path}{RESET}")
                except Exception as e:
                    print(f"{YELLOW}Could not start Stockfish ({e}); continuing "
                          f"without eval.{RESET}")

    def close(self):
        if self.evaluator:
            self.evaluator.close()

    # ── rules / state ────────────────────────────────────────────────────
    def initial_state(self):
        return {"board": chess.Board(), "comments": [], "last": None,
                "eval_loss": {"white": [], "black": []}}

    def current_role(self, state):
        return "white" if state["board"].turn == chess.WHITE else "black"

    def system_prompt(self, role):
        return SYSTEM_TMPL.format(color=role.capitalize())

    def observation(self, state, role):
        board = state["board"]
        parts = [
            f"You are {role.capitalize()}. It is your move "
            f"(move {board.fullmove_number}).",
            f"\nMove history (SAN):\n{san_history(board)}",
            f"\nFEN:\n{board.fen()}",
        ]
        if self.board_input:
            parts.append("\nBoard (uppercase = White, lowercase = Black, . = empty):\n"
                         + board_letters(board))
        parts.append("\nGive your move now as JSON.")
        return "\n".join(parts)

    def action_schema(self, state, role):
        return MOVE_SCHEMA

    def action_summary(self, action):
        return str(action.get("move", "?"))

    def apply(self, state, role, action):
        board = state["board"]
        move, ambiguous = parse_move(board, action.get("move", ""))
        if move is None and ambiguous:
            # Name the piece and square but NOT the candidate notations — the
            # model must disambiguate its own intent, not pick from a list.
            mv0 = board.parse_san(ambiguous[0])
            piece_name = chess.piece_name(board.piece_type_at(mv0.from_square))
            to_sq = chess.square_name(mv0.to_square)
            return "illegal", (f"AMBIGUOUS — more than one of your {piece_name}s "
                               f"can reach {to_sq}, so SAN requires you to say "
                               "which one; the move itself may be fine once "
                               "disambiguated")
        if move is None:
            return "illegal", (f'"{action.get("move", "")}" is not a legal move '
                               "in this position")
        if self.evaluator is not None:
            loss = self.evaluator.loss(board, move)
            if loss is not None:
                state["eval_loss"][role].append(loss)
        san = board.san(move)
        board.push(move)
        state["last"] = move
        state["comments"].append(str(action.get("comment", "")).strip())
        check = ""
        if board.is_checkmate():
            check = f" {RED}{BOLD}#{RESET}"
        elif board.is_check():
            check = f" {YELLOW}+{RESET}"
        return "ok", f"{san}{check}"

    def is_over(self, state):
        return state["board"].is_game_over(claim_draw=True)

    # ── presentation / results ───────────────────────────────────────────
    def render(self, state):
        return render_board(state["board"], last_move=state["last"])

    def result(self, state, forfeit=None, capped=False):
        board = state["board"]
        if forfeit is not None:
            role, kind = forfeit
            result_str = "0-1" if role == "white" else "1-0"
            why = "ran out of time" if kind == "time" else "illegal move"
            summary = f"forfeit ({role.capitalize()} {why})"
        elif capped:
            diff = material_diff(board)
            if abs(diff) >= 3:
                result_str = "1-0" if diff > 0 else "0-1"
            else:
                result_str = "1/2-1/2"
            summary = "adjudicated at ply cap"
        else:
            outcome = board.outcome(claim_draw=True)
            result_str = outcome.result()
            summary = str(outcome.termination).split(".")[-1].lower()

        winner = {"1-0": "white", "0-1": "black", "1/2-1/2": None}[result_str]
        res = {
            "winner": winner, "summary": summary, "result": result_str,
            "scores": {"white": {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[result_str],
                       "black": {"1-0": 0.0, "0-1": 1.0, "1/2-1/2": 0.5}[result_str]},
            "plies": board.ply(),
        }
        if self.evaluator is not None:
            res["acpl"] = {r: (round(sum(v) / len(v), 1) if v else None)
                           for r, v in state["eval_loss"].items()}
        return res

    def export(self, state, outcome, run_dir):
        """Replayable PGN with each model's explanation as a move comment."""
        board = state["board"]
        game = chess.pgn.Game.from_board(board)
        game.headers["Event"] = "AI Games chess"
        game.headers["Site"] = "ollama"
        game.headers["Date"] = date.today().strftime("%Y.%m.%d")
        game.headers["White"] = outcome["labels"]["white"]
        game.headers["Black"] = outcome["labels"]["black"]
        game.headers["WhiteModel"] = outcome["models"]["white"]
        game.headers["BlackModel"] = outcome["models"]["black"]
        game.headers["Result"] = outcome.get("result", "*")
        game.headers["Termination"] = outcome.get("summary", "")

        node = game
        for comment in state["comments"]:
            node = node.variations[0] if node.variations else node
            if comment:
                node.comment = comment

        path = os.path.join(run_dir, "game.pgn")
        with open(path, "w", encoding="utf-8") as f:
            print(game, file=f, end="\n")
        return [path]
