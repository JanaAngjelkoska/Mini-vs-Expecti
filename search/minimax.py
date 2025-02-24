from eval.evaluation import Evaluator
from utility.ordering import Ordering
from chess import *
from typing import Tuple, Optional

import numpy as np


class Minimax:
    """
        The minimax adversarial search algorithm.
    """

    # A Transposition Table, used to memoize evaluations of positions we have already seen
    TT = dict()

    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator

    def search(self, cur_depth: int, max_depth: int, white_turn: bool, alpha: float, beta: float, board: Board) \
            -> Tuple[float, Optional[Move]]:
        """
            A function allowing the minimax algorithm to calculate a move to play.

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

        if board.legal_moves.count() == 0:
            if board.is_check():  # checkmate
                return -np.inf if white_turn else np.inf, None
            else:  # stalemate
                return 0, None

        fen_parts = board.fen().split(' ')
        fen_hash = ' '.join(fen_parts[:4])

        if fen_hash in Minimax.TT:
            tt_eval, tt_depth, tt_bm = Minimax.TT[fen_hash]
            if cur_depth >= tt_depth:
                return tt_eval, tt_bm

        if cur_depth == max_depth:
            return self.evaluator.evaluate_position(board), None

        best_move = None

        ordered_moves = Ordering.order(board.legal_moves, board, cur_depth)

        if white_turn:
            max_eval = -np.inf

            for lm in ordered_moves:
                board.push(lm)

                if board.is_checkmate():
                    board.pop()
                    return np.inf, lm

                score, _ = self.search(cur_depth + 1, max_depth, False, alpha, beta, board)
                board.pop()
                if score > max_eval:
                    max_eval = score
                    best_move = lm

                alpha = max(alpha, score)
                if beta <= alpha:
                    Ordering.push_killer(lm, cur_depth)
                    break

            Ordering.update_history_heuristic(best_move, cur_depth)
            Minimax.cache_result(fen_hash, max_eval, best_move, cur_depth)

            return max_eval, best_move

        else:
            min_eval = np.inf

            for lm in ordered_moves:
                board.push(lm)

                if board.is_checkmate():
                    board.pop()
                    return -np.inf, lm

                score, _ = self.search(cur_depth + 1, max_depth, True, alpha, beta, board)
                board.pop()
                if score < min_eval:
                    min_eval = score
                    best_move = lm
                beta = min(beta, score)
                if beta <= alpha:
                    Ordering.push_killer(lm, cur_depth)
                    break

            Ordering.update_history_heuristic(best_move, cur_depth)
            Minimax.cache_result(fen_hash, min_eval, best_move, cur_depth)

            return min_eval, best_move

    @staticmethod
    def cache_result(key: str, ev: float, move: Move, dep: int) -> None:
        """
            Helper method that provides an interface to effectively cache a currently-evaluated position.

            Arguments:
            :param key: The key that will hash board state information.
            :param ev: The score evaluated at this position.
            :param move: The move to be played, that is in accordance with the score.
            :param dep: The depth at which this position is evaluated.
        """

        if key in Minimax.TT:
            _, tt_depth, _ = Minimax.TT[key]
            if dep <= tt_depth:
                return

        Minimax.TT[key] = (ev, dep, move)
