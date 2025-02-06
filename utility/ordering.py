from chess import *

class Ordering:
    """
    A specific move ordering used to prune the search tree earlier (prioritize better moves).
    """

    PIECE_VALUES = {
        None: 0,
        PAWN: 100,
        KNIGHT: 300,
        BISHOP: 300,
        ROOK: 500,
        QUEEN: 900,
        KING: 99999999,
    }

    history_heuristic = {}
    killer_moves = {}

    @staticmethod
    def update_history_heuristic(move, depth):
        if move not in Ordering.history_heuristic:
            Ordering.history_heuristic[move] = 0
        Ordering.history_heuristic[move] += 2 ** depth

    @staticmethod
    def get_killer_moves(cur_depth):
        return Ordering.killer_moves.get(cur_depth, [])

    @staticmethod
    def add_killer_move(move, cur_depth):
        if cur_depth not in Ordering.killer_moves:
            Ordering.killer_moves[cur_depth] = []
        Ordering.killer_moves[cur_depth].append(move)

    @staticmethod
    def mvv_lva(move: Move, board: Board) -> int:
        """
        Calculate the Most Valuable Victim - Least Valuable Attacker (MVV - LVA) score for a given move.
        """
        attacker_value = Ordering.PIECE_VALUES[board.piece_at(move.from_square).piece_type]
        victim_piece = board.piece_at(move.to_square)
        if victim_piece:
            victim_value = Ordering.PIECE_VALUES[victim_piece.piece_type]
        else:
            victim_value = 0
        return victim_value * 10 - attacker_value

    @staticmethod
    def order(legal_moves: LegalMoveGenerator, board: Board, cur_depth: int) -> list:
        """
        Order moves using the Most Valuable Victim - Least Valuable Attacker (MVV - LVA) heuristic,
        with killer moves and history heuristic prioritized.
        """
        moves = list(legal_moves)
        killer_moves = Ordering.get_killer_moves(cur_depth)

        # Separate killer moves and non-killer moves
        killer_moves_set = set(killer_moves)
        killer_moves_list = [move for move in moves if move in killer_moves_set]
        non_killer_moves_list = [move for move in moves if move not in killer_moves_set]

        non_killer_moves_list.sort(
            key=lambda move: (Ordering.history_heuristic.get(move, 0), Ordering.mvv_lva(move, board)),
            reverse=True  # desc
        )

        return killer_moves_list + non_killer_moves_list

