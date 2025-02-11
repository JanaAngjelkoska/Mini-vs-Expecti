import numpy as np

from abc import ABC, abstractmethod
from chess import *

class Heuristic(ABC):
    """
        Functional interface to implement for a heuristic.
    """

    @abstractmethod
    def estimate(self, board: Board, color: Color) -> float:
        """
            A function that estimates the position, given 1 heuristic to estimate on.

            Arguments:
            :param board: An arbitrary board state.
            :param color: The color for which the evaluation is executed.
            :return: An integer estimate of the position given the heuristic.
        """
        pass

    @staticmethod
    def has_pawns_on_cols(board: Board, color: Color) -> np.array:
        columns = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}

        pawns = board.pieces(PAWN, color)

        contains_pawn = np.zeros(8)

        for pawn in pawns:
            column = square_name(pawn)[0]
            at_position = columns[column]
            contains_pawn[at_position] = 1

        return contains_pawn


class DoubledPawns(Heuristic):
    """
        Heuristic for calculating a doubled pawns penalty.
    """

    def estimate(self, board: Board, color: Color) -> float:
        pawns = board.pieces(PAWN, color)

        columns = -np.ones(8)

        for pawn in pawns:
            column = ord(square_name(pawn)[0]) - ord('a')
            columns[column] += 0.7

        return -np.sum(columns)


class IsolatedPawns(Heuristic):
    """
        Heuristic for calculating an isolated pawns/pawn island penalty.
    """

    def estimate(self, board: Board, color: Color) -> float:
        contains_pawn = Heuristic.has_pawns_on_cols(board, color)

        isolated = 0

        for pos in range(8):

            if pos == 0 and contains_pawn[pos] == 1 and contains_pawn[pos + 1] == 0:
                isolated += 1
                continue

            if pos == 7 and contains_pawn[pos] == 1 and contains_pawn[pos - 1] == 0:
                isolated += 1
                continue

            if contains_pawn[pos] == 1 and contains_pawn[pos - 1] == 0 and contains_pawn[pos + 1] == 0:
                isolated += 1
                continue

        return -isolated


class PassedPawns(Heuristic):
    """
        Heuristic for calculating a passed pawns reward.
    """

    def estimate(self, board: Board, color: Color) -> float:

        adversary_color = not color

        has_pawn_for_col = Heuristic.has_pawns_on_cols(board, color)
        has_pawn_against_col = Heuristic.has_pawns_on_cols(board, adversary_color)

        passed = PassedPawns.__check_passed(check_for=has_pawn_for_col, check_against=has_pawn_against_col)

        return passed * 1.5

    @staticmethod
    def __check_passed(check_for: np.ndarray[int], check_against: np.ndarray[int]) -> float:
        """
            Helper function for detecting pawn presence (bool -> there or not there) in columns.

            Arguments:
            :param check_for: Indexed array of columns (0->a, 1->b, ..., 7->h) of the current player's pawn presence.
            :param check_against: Indexed array of columns (0->a, 1->b, ..., 7->h)
                                    of the adversary's pawn presence.
            :return: Number of current player's passwed pawns.
        """
        has_passed = 0

        if check_against[0] == 0 and check_against[1] == 0 and check_for[0] == 1:
            has_passed += 1
        if check_against[7] == 0 and check_against[6] == 0 and check_for[7] == 1:
            has_passed += 1

        for pawn in range(1, 7):
            if check_for[pawn] == 0:
                continue

            if check_against[pawn] == 0 and check_against[pawn - 1] == 0 and check_against[pawn + 1] == 0:
                has_passed += 1

        return has_passed


class BishopAttacks(Heuristic):
    """
        Heuristic for detecting how much control a bishop has over opponents.
    """

    def estimate(self, board: Board, color: Color) -> float:
        bishop_squares = board.pieces(BISHOP, color)

        adversary_color = not color

        score = 0

        for sq in bishop_squares:
            attacked_squares = board.attacks(sq)
            score += sum(
                [1
                 for sq in attacked_squares
                 if board.piece_at(sq) is not None and board.piece_at(sq).color == adversary_color]
            )

        return score * 0.6


class BishopPair(Heuristic):
    """
        Heuristic that detects if the Bishop pair is present for the current player.
    """

    def estimate(self, board: Board, color: Color) -> float:
        return 2 if len(board.pieces(BISHOP, color)) == 2 else 0


class KingSafety(Heuristic):
    """
        Heuristic that detects how safe the current player's king is, based on how many squares around the king are attacked.
    """

    def estimate(self, board: Board, color: Color) -> float:
        king_color = color

        adversary_color = not color

        attacked_squares = []

        king_squares = board.pieces(KING, king_color)

        king_square = None

        for sq in king_squares:
            king_square = sq

        for sq in SQUARES:
            if square_distance(king_square, sq) <= 1:
                attacked_squares.append(-1 * len(board.attackers(adversary_color, sq)))

        return sum(attacked_squares)


class OpenRook(Heuristic):

    def estimate(self, board: Board, color: Color) -> float:
        rooks = board.pieces(ROOK, color)
        rooks_squares = [square_name(sq) for sq in rooks]

        pawns = board.pieces(PAWN, color)
        pawn_squares = [square_name(sq) for sq in pawns]

        open_rooks = dict()

        for rook in rooks_squares:
            is_open = True
            for pawn in pawn_squares:
                rook_file = rook[0]
                pawn_file = pawn[0]

                rook_row = rook[1]
                pawn_row = pawn[1]

                if rook_file == pawn_file and pawn_row > rook_row:
                    is_open = False
                    break

            open_rooks[rook] = 1 if is_open else 0

        return sum(open_rooks.values())


class PieceMobility(Heuristic):
    def estimate(self, board: Board, color: Color) -> float:
        cp = Board(board.fen())
        cp.turn = color
        return len(list(board.legal_moves))


class Material(Heuristic):
    def estimate(self, board: Board, color: Color) -> float:

        from eval.evaluation import Evaluator

        piece_values = {
            PAWN: 1,
            KNIGHT: 3,
            BISHOP: 3,
            ROOK: 5,
            QUEEN: 9,
        }

        material_count = 0

        for piece_type in piece_values.keys():
            material_count += Evaluator.piece_presence[color][piece_type] * piece_values[piece_type]

        return material_count


class CenterControl(Heuristic):
    def estimate(self, board: Board, color: Color) -> float:
        central_squares = {E4, E5, D4, D5}
        center_grasp_score = 0
        pawns = board.pieces(PAWN, color)

        for sq in central_squares:
            attackers = board.attackers(color, sq)
            center_grasp_score += 0.5 * len(attackers)

        # higher reward if there are pawns on the central squares
        for pawn in pawns:
            if pawn in central_squares:
                center_grasp_score += 1

        return center_grasp_score * 0.75


class EarlyQueenPenalty(Heuristic):

    def estimate(self, board: Board, color: Color) -> float:

        from eval.evaluation import Evaluator

        queen_square = D1 if color == WHITE else D8

        queens = board.pieces(QUEEN, color)

        if Evaluator.move_no > 7:
            return 0

        curr_q_sq = None

        for q in queens:
            curr_q_sq = q

        if curr_q_sq != queen_square:
            return -5

        return 0


class EarlyKingPenalty(Heuristic):

    def estimate(self, board: Board, color: Color) -> float:

        from eval.evaluation import Evaluator

        king_square = D1 if color == WHITE else D8

        kings = board.pieces(KING, color)

        if Evaluator.move_no > 7:
            return 0

        curr_k_sq = None

        for q in kings:
            curr_k_sq = q

        if square_distance(curr_k_sq, king_square) == 1:
            return -10

        return 0


class PieceInactivity(Heuristic):

    def estimate(self, board: Board, color: Color) -> float:
        initial_board = Board()
        count = 0

        for sq in SQUARES:
            current_piece = board.piece_at(sq)
            initial_piece = initial_board.piece_at(sq)

            if current_piece == initial_piece and current_piece is not None and current_piece.color == color:
                count += 1

        return count * -1


# class Checkmate(Heuristic):
#     def estimate(self, board: Board, color: Color) -> float:
#         if board.is_checkmate():
#             if board.turn == color:
#                 return float('-inf')
#             else:
#                 return float('inf')
#         return 0


class WeakAttackers(Heuristic):

    def estimate(self, board: Board, color: Color) -> float:
    # todo za jana: test
        pieces = {
            PAWN: 1,
            KNIGHT: 3,
            BISHOP: 3,
            ROOK: 5,
            QUEEN: 9,
        }

        legal_moves = board.legal_moves

        attacker_moves = [move for move in legal_moves]
        attacker_squares = [move.to_square for move in attacker_moves]
        all_pieces_to_check = board.occupied_co[color]

        attacked_by = 0

        for move, sq in zip(attacker_moves, attacker_squares):
            board.push(move)
            attacker_piece = board.piece_at(sq)

            attacked_squares = board.attacks(sq)
            attacked_pieces = SquareSet(SquareSet(all_pieces_to_check) & attacked_squares)

            for attacked_square in attacked_pieces:
                attacked_piece_at_square = board.piece_at(attacked_square)

                if attacked_piece_at_square is None:
                    continue

                if (
                    pieces.get(attacker_piece.piece_type, 0)
                        <
                    pieces.get(attacked_piece_at_square.piece_type, 0)
                ):
                    attacked_by += 1

            board.pop()

        return -attacked_by
