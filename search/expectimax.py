from chess import Board, Move
from typing import Tuple, Optional
import numpy as np
from eval.evaluation import Evaluator


class Expectiminimax:
    def __init__(self, eval_engine: Evaluator):
        self.eval_engine = eval_engine

    def execute(self, board, depth: int, is_max_player: bool) -> Tuple[float, Optional[Move]]:

        if depth == 0:
            return self.eval_engine.evaluate_position(board), None

        legal_moves = list(board.legal_moves)

        if is_max_player:
            best_value = -np.inf
            best_move = None
            for move in legal_moves:
                board.push(move)
                evaluation, _ = self.execute(board, depth - 1, False)
                board.pop()

                if evaluation > best_value:
                    best_value = evaluation
                    best_move = move

            return best_value, best_move

        elif not is_max_player:
            best_value = np.inf
            best_move = None
            for move in legal_moves:
                board.push(move)
                evaluation, _ = self.execute(board, depth - 1, True)
                board.pop()

                if evaluation < best_value:
                    best_value = evaluation
                    best_move = move

            return best_value, best_move

        else:  # ova kako da go impl?
            expected_value = 0
            probabilities = [1.0 / len(legal_moves) for _ in legal_moves]

            for move, prob in zip(legal_moves, probabilities):
                board.push(move)
                evaluation, _ = self.execute(board, depth - 1, True)
                board.pop()
                expected_value += prob * evaluation

            return expected_value, None


# eval_engine = EvaluationEngine()
# board = Board()
# root_node = ExpectimaxNode(board, parent_type="max")
# expectiminimax = Expectiminimax(eval_engine)
# depth = 3
# best_value, best_move = expectiminimax.execute(root_node, depth, is_max_player=True)
# print(f"Best move: {best_move}, Evaluation value: {best_value}")
