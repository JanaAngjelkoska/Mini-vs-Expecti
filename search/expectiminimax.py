from eval.evaluation import Evaluator
from chess import *
from utility.ordering import Ordering
import numpy as np


class Expectiminimax:
    """
        The Expectiminimax algorithm, extending Minimax to handle chance nodes.
    """

    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator

    def search(self, cur_depth: int, max_depth: int, white_turn: bool, board: Board, alpha: float = -np.inf,
               beta: float = np.inf) -> Tuple[float, Optional[Move]]:
        """
            Expectiminimax search with artificially introduced chance nodes and alpha-beta pruning for MAX and MIN nodes.
        """

        if board.legal_moves.count() == 0:
            if board.is_check():  # checkmate
                return -np.inf if white_turn else np.inf, None
            else:  # stalemate
                return 0, None

        if cur_depth == max_depth:
            return self.evaluator.evaluate_position(board), None

        # agent plays move from chance node
        if cur_depth % 2 == 0:
            return self.__chance_node(cur_depth, max_depth, white_turn, board)

        ordered_moves = Ordering.order(board.legal_moves, board, cur_depth)

        # MAX or MIN
        best_move = None
        if white_turn:
            max_eval = -np.inf
            for lm in ordered_moves:
                board.push(lm)

                if board.is_checkmate():
                    return -np.inf if white_turn else np.inf, lm

                score, _ = self.search(cur_depth + 1, max_depth, not white_turn, board, alpha, beta)
                board.pop()

                if score > max_eval:
                    max_eval = score
                    best_move = lm

                # Alpha-beta pruning
                alpha = max(alpha, max_eval)
                if beta <= alpha:
                    Ordering.push_killer(best_move, cur_depth)
                    break

            Ordering.update_history_heuristic(best_move, cur_depth)

            return max_eval, best_move

        else:
            min_eval = np.inf
            for lm in board.legal_moves:
                board.push(lm)
                score, _ = self.search(cur_depth + 1, max_depth, not white_turn, board, alpha, beta)
                board.pop()

                if score < min_eval:
                    min_eval = score
                    best_move = lm

                beta = min(beta, min_eval)
                if beta <= alpha:
                    Ordering.push_killer(best_move, cur_depth)
                    break

            Ordering.update_history_heuristic(best_move, cur_depth)

            return min_eval, best_move

    def __chance_node(self, cur_depth: int, max_depth: int, white_turn: bool, board: Board) -> Tuple[float, None]:
        """
            Simulates a chance node using a random probability distribution,
            minimizing expected value if it's Black's turn, maximizing if it's White's turn.
        """
        EScore = -np.inf if white_turn else np.inf

        num_moves = board.legal_moves.count()

        random_vec = np.random.uniform(size=num_moves)
        random_vec /= np.sum(random_vec)  # normalize to generate a random probability dist.

        lm_likelihood_best = None

        for i, lm in enumerate(board.legal_moves):
            # Evaluator.piececount_update(board, lm)
            board.push(lm)
            score, _ = self.search(cur_depth + 1, max_depth, not white_turn, board)
            board.pop()
            # Evaluator.pop_upd_stack()

            # Update EScore based on whether it's white or black's turn
            ev_score = score * random_vec[i]

            if white_turn:
                if ev_score > EScore:
                    EScore = score
                    lm_likelihood_best = lm
            else:
                if ev_score < EScore:
                    EScore = score
                    lm_likelihood_best = lm

        return EScore, lm_likelihood_best
