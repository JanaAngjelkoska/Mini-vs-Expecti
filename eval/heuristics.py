from abc import ABC, abstractmethod

from chess import *
from sklearn.preprocessing import MinMaxScaler

import numpy as np


class Heuristic(ABC):
    """
    Functional interface to implement for a heuristic.
    """

    @abstractmethod
    def estimate(self, board: Board, color: Color) -> float:
        """
            A function that estimates the position, given 1 heuristic to estimate on.
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
            columns[column] += 1

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

        return isolated * -1


class PassedPawns(Heuristic):
    """
        Heuristic for calculating a passed pawns reward.
    """
    def estimate(self, board: Board, color: Color) -> float:

        adversary_color = not color

        has_pawn_for_col = Heuristic.has_pawns_on_cols(board, color)
        has_pawn_against_col = Heuristic.has_pawns_on_cols(board, adversary_color)

        passed = PassedPawns.__check_passed(check_for=has_pawn_for_col, check_against=has_pawn_against_col)

        return passed * 1

    @staticmethod
    def __check_passed(check_for: np.ndarray[int], check_against: np.ndarray[int]) -> float:
        """
        Helper function for detecting pawn presence (bool -> there or not there) in columns.
            :param check_for: Indexed array of columns (0 -> a, 1 -> b, ..., 7 -> h) of the current player's pawn presence.
            :param check_against: Indexed array of columns (0 -> a, 1 -> b, ..., 7 -> h)
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

        return score


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

class Checkmate(Heuristic):

    def estimate(self, board: Board, color: Color) -> float:
        return np.inf if board.is_checkmate() else 0

class Stalemate(Heuristic):

    def estimate(self, board: Board, color: Color) -> float:
        return 0 if board.is_stalemate() else -np.inf

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

        return sum(open_rooks.values()) * 1

class PieceMobility(Heuristic):
    def estimate(self, board: Board, color: Color) -> float:
        copy = Board(board.fen())
        copy.turn = color
        return len(list(board.legal_moves)) * 1

class Material(Heuristic):
    def estimate(self, board: Board, color: Color) -> float:

        piece_values = {
            PAWN:   1,
            KNIGHT: 3,
            BISHOP: 3,
            ROOK:   5,
            QUEEN:  9,
        }

        material_count = 0

        for piece_type in piece_values.keys():
            white_pieces = board.pieces(piece_type, color)

            material_count += len(white_pieces) * piece_values[piece_type]

        return material_count

class CenterControl(Heuristic):
    def estimate(self, board: Board, color: Color) -> float:
        central_squares = SquareSet([E4, E5, D4, D5])

        center_grasp_score = 0

        for sq in central_squares:
            attackers = board.attackers(color, sq)
            center_grasp_score += 1 * len(attackers)

        return center_grasp_score

class EarlyQueenPenalty(Heuristic):

    def estimate(self, board: Board, color: Color) -> float:

        queen_square = D1 if color == WHITE else D8

        queens = board.pieces(QUEEN, color)

        if EvaluationEngine.move_no > 7:
            return 0

        curr_q_sq = None

        for q in queens:
            curr_q_sq = q

        if curr_q_sq != queen_square:
            return -5

        return 0

class PieceInactivity(Heuristic):

    def estimate(self, board: Board, color: Color) -> float:
        initial_board = Board()
        count = 0

        for sq in SQUARES:
            current_piece = board.piece_at(sq)
            initial_piece = initial_board.piece_at(sq)

            # Check if the current piece matches the initial piece and is of the specified color
            if current_piece == initial_piece and current_piece is not None and current_piece.color == color:
                count += 1

        return count * -1

class EvaluationEngine:

    move_no = 1

    weight_map = {
        Material: 0.4,
        PassedPawns: 0.07,
        KingSafety: 0.12,
        EarlyQueenPenalty: 0.05,
        PieceMobility: 0.08,
        CenterControl: 0.07,
        BishopPair: 0.03,
        OpenRook: 0.02,
        BishopAttacks: 0.03,
        DoubledPawns: 0.02,
        IsolatedPawns: 0.02,
        PieceInactivity: 0.09,
        Checkmate: 1.0,
        Stalemate: 0.0,
    }

    transposition_table = dict()

    def __init__(self, *heuristics):
        """
        :param heuristics: Object members of the heuristic class, used to cumulatively calculate the final position
        evaluation. Accepts n heuristic objects.
        """
        self.heuristics_use = []
        self.weights = None

        all_heuristics = Heuristic.__subclasses__()
        all_heuristics.remove(Stalemate)

        if not heuristics:
            self.heuristics_use = list(map(lambda h: h(), all_heuristics))
        else:
            self.heuristics_use = [*heuristics]
            self.heuristics_use += [Checkmate()]

        self.weightvec = self.__build_weight_vector()

    def __build_weight_vector(self) -> np.array:
        """
            Helper function to build the weight vector. If n heuristic objects are passed, a weight vector of R^n is
            constructed according to the weight importances of the field weight_map.
            :return: A weight vector of R^n.
        """
        weight_vector = np.empty_like(self.heuristics_use)
        for i, heuristic in enumerate(self.heuristics_use):
            weight_vector[i] = EvaluationEngine.weight_map[type(heuristic)]

        update_indices = []

        for ind, w in enumerate(weight_vector):
            if w != 0.9:
                update_indices.append(ind)

        to_scale = weight_vector[update_indices]

        scaler = MinMaxScaler(feature_range=(min(to_scale), max(to_scale)))
        scaled = scaler.fit_transform(to_scale.reshape(-1, 1)).flatten()
        scaled = scaled / np.sum(scaled)

        weight_vector[update_indices] = scaled

        return weight_vector

    def evaluate_position(self, board: Board) -> float:
        """
        Evaluation of any board position/state. A positive result is an advantage for white, the converse is an
        advantage for black. A win for white is given by np.inf, while -np.inf is a win for black.
        A zero evalution is a draw.
            :param board: An arbitrary board state.
            :return: Inner product of the evaluation for white minus the inner product of the evaluation for black.
        """

        stale_score = Stalemate().estimate(board, board.turn)

        if stale_score == 0:
            return stale_score

        evaluation_vec_white = np.array(
            list(map(lambda h: h.estimate(board, WHITE), self.heuristics_use))
        )

        white_score = np.dot(self.weightvec, evaluation_vec_white)

        if white_score == np.inf:
            checkmate_info = np.inf if board.turn == BLACK else -np.inf
            return checkmate_info

        evaluation_vec_black = np.array(
            list(map(lambda h: h.estimate(board, BLACK), self.heuristics_use))
        )

        black_score = np.dot(self.weightvec, evaluation_vec_black)

        return white_score - black_score

