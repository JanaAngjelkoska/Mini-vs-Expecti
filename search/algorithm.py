import time

from eval.heuristics import *
from utility.ordering import Ordering
from chess import *
from typing import Tuple, Optional
import numpy as np


class Algorithm(ABC):
    """
        An interface to implement to define a specific adversarial search algorithm.
    """

    def __init__(self, eval_engine: EvaluationEngine):
        self.eval_engine = eval_engine

    @abstractmethod
    def execute(self, cur_depth: int, max_depth: int, white_turn: bool, alpha: Optional[float], beta: Optional[float],
                board: Board) -> Tuple[float, Optional[Move]]:
        """
            A function allowing an adversarial search algorithm to calculate a move to play.

            Arguments:
            :param cur_depth: The depth that the algorithm starts from.
            :param max_depth: The depth that the algorithm searches until.
            :param white_turn: The current player's turn.
            :param alpha: Parameter used by the maximizer (player wanting to maximize the score - White). Used
                          primarily in Minimax optimization and NOT meant to be used directly in other algorithms.
            :param beta: Parameter used by the minimizer (player wanting to minimize the score - Black). Used
                          primarily in Minimax optimization and NOT meant to be used directly in other algorithms.
            :param board: A board state that we are starting to search from.
            :return: A tuple, the first element being the evaluation estimate, the second being the move to play next.
        """
        pass


class Minimax(Algorithm):

    TT = dict()

    def __init__(self, eval_engine: EvaluationEngine):
        super().__init__(eval_engine)

    def execute(self, cur_depth: int, max_depth: int, white_turn: bool, alpha: float, beta: float, board: Board) \
        -> Tuple[float, Optional[Move]]:

        if board.legal_moves.count() == 0:
            if board.is_check():  # checkmate
                return -np.inf if white_turn else np.inf, None
            else:  # stalemate
                return 0, None

        fen_parts = board.fen().split(' ')
        fen_hash = ' '.join(fen_parts[:4])

        if fen_hash in EvaluationEngine.transposition_table:
            score, depth, best_move = Minimax.TT[fen_hash]
            if cur_depth <= depth:
                return score, best_move

        if cur_depth == max_depth:
            return self.eval_engine.evaluate_position(board), None

        best_move = None

        if white_turn:
            max_eval = -np.inf
            ordered_moves = Ordering.order(board.legal_moves, board, cur_depth)

            for lm in ordered_moves:
                EvaluationEngine.update_piececounts_after(board, lm)
                board.push(lm)
                score, _ = self.execute(cur_depth + 1, max_depth, False, alpha, beta, board)
                board.pop()
                EvaluationEngine.undo_piececounts_last()
                if score > max_eval:
                    max_eval = score
                    best_move = lm
                alpha = max(alpha, score)
                if beta <= alpha:
                    Ordering.add_killer_move(lm, cur_depth)
                    break

            Ordering.update_history_heuristic(best_move, cur_depth)
            Minimax.cache_result(fen_hash, max_eval, best_move, cur_depth)

            return max_eval, best_move

        else:
            min_eval = np.inf
            ordered_moves = Ordering.order(board.legal_moves, board, cur_depth)

            for lm in ordered_moves:
                EvaluationEngine.update_piececounts_after(board, lm)
                board.push(lm)
                score, _ = self.execute(cur_depth + 1, max_depth, True, alpha, beta, board)
                board.pop()
                EvaluationEngine.undo_piececounts_last()
                if score < min_eval:
                    min_eval = score
                    best_move = lm
                beta = min(beta, score)
                if beta <= alpha:
                    Ordering.add_killer_move(lm, cur_depth)
                    break

            Ordering.update_history_heuristic(best_move, cur_depth)
            Minimax.cache_result(fen_hash, min_eval, best_move, cur_depth)

            return min_eval, best_move

    @staticmethod
    def cache_result(key, ev, move, dep):
        if key in Minimax.TT:
            tt_eval, tt_depth, tt_best_move = Minimax.TT[key]
            if dep < tt_depth:
                return

        Minimax.TT[key] = (ev, dep, move)


# Example Usage - todo: remove in future
board_init = Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')

algo = Minimax(EvaluationEngine())

while not board_init.is_checkmate() and not board_init.is_stalemate():
    print('% % % % % % % %')
    print(board_init)
    print('% % % % % % % %\n')

    # User's turn
    while True:
        start = time.time()
        m = input("Your move (in SAN): ").strip()
        print(f"You took {time.time() - start:.2f}s")
        try:
            EvaluationEngine.update_piececounts_after(board_init, Move.from_uci(m))
            board_init.push_san(m)
            break
        except ValueError:
            print("Invalid move.")

    if board_init.is_checkmate():
        print("Checkmate for white.")
        break
    elif board_init.is_stalemate():
        print("Stalemate.")
        break

    print("Algorithm's turn...")
    start = time.time()
    evaluation, m = algo.execute(0, 4, False, -np.inf, np.inf, board_init)
    print(len(EvaluationEngine.transposition_table.keys()))
    print(m)
    EvaluationEngine.update_piececounts_after(board_init, m)
    board_init.push(m)
    print(f"Opponent took {time.time() - start:.2f}s")

    if board_init.is_checkmate():
        print("Checkmate for black.")
        break
    elif board_init.is_stalemate():
        print("Draw.")
        break

    print(f"Evaluation: {evaluation:.2f}")
    EvaluationEngine.move_no += 1
