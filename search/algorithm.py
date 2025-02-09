import time

from eval.heuristics import *
from utility.ordering import Ordering


class Algorithm(ABC):
    """
        An interface to implement to define a specific adversarial search algorithm.
    """

    def __init__(self, eval_engine: EvaluationEngine):
        self.eval_engine = eval_engine
        pass

    @abstractmethod
    def execute(self, cur_depth: int, max_depth: int, white_turn: bool, alpha: float, beta: float, board: Board) \
            -> Tuple[float, Optional[Move], Board]:
        pass


class Minimax(Algorithm):

    def __init__(self, eval_engine: EvaluationEngine):
        super().__init__(eval_engine)

    def execute(self, cur_depth: int, max_depth: int, white_turn: bool, alpha: float, beta: float, board: Board) -> \
            Tuple[float, Optional[Move]]:

        fen_parts = board.fen().split(' ')
        relevant_fen = ' '.join(fen_parts[:4])

        if board.is_checkmate():
            return (np.inf if white_turn else -np.inf), None

        if relevant_fen in EvaluationEngine.transposition_table:
            score, depth, best_move = EvaluationEngine.transposition_table[relevant_fen]
            if cur_depth <= depth:
                return score, best_move

        if cur_depth == max_depth or board.is_stalemate():
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
            if relevant_fen in EvaluationEngine.transposition_table:
                tt_eval, tt_depth, tt_best_move = EvaluationEngine.transposition_table[relevant_fen]
                if cur_depth >= tt_depth:
                    EvaluationEngine.transposition_table[relevant_fen] = (max_eval, cur_depth, best_move)

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
            if relevant_fen in EvaluationEngine.transposition_table:
                tt_eval, tt_depth, tt_best_move = EvaluationEngine.transposition_table[relevant_fen]
                if cur_depth >= tt_depth:
                    EvaluationEngine.transposition_table[relevant_fen] = (min_eval, cur_depth, best_move)

            return min_eval, best_move


# Example Usage - todo: remove in future
board = Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')

algo = Minimax(EvaluationEngine())

while not board.is_checkmate() and not board.is_stalemate():
    print('% % % % % % % %')
    print(board)
    print('% % % % % % % %\n')

    # User's turn
    while True:
        start = time.time()
        move = input("Your move (in SAN): ").strip()
        print(f"You took {time.time() - start:.2f}s")
        try:
            EvaluationEngine.update_piececounts_after(board, Move.from_uci(move))
            board.push_san(move)
            break
        except ValueError:
            print("Invalid move.")

    if board.is_checkmate():
        print("Checkmate for white.")
        break
    elif board.is_stalemate():
        print("Stalemate.")
        break

    print("Algorithm's turn...")
    start = time.time()
    evalscore, move = algo.execute(0, 4, False, -np.inf, np.inf, board)
    print(len(EvaluationEngine.transposition_table.keys()))
    print(move)
    EvaluationEngine.update_piececounts_after(board, move)
    board.push(move)
    print(f"Opponent took {time.time() - start:.2f}s")

    if board.is_checkmate():
        print("Checkmate for black.")
        break
    elif board.is_stalemate():
        print("Draw.")
        break

    print(f"Evaluation: {evalscore:.2f}")
    EvaluationEngine.move_no += 1
