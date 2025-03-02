from eval.heuristics import (Heuristic, Material, PassedPawns, DoubledPawns, IsolatedPawns, BishopPair, BishopAttacks,
                             CenterControl, KingSafety, EarlyKingPenalty, EarlyQueenPenalty, WeakAttackers, OpenRookFile,
                             PieceMobility, PieceInactivity)
from chess import *

import numpy as np


class Evaluator:
    # === Static Attributes === #

    # move number tracker
    move_no = 1

    # Weight map for opening
    weight_map_open = {
        # todo: make weights better :D :p
        Material: 1,
        PassedPawns: 0.01,
        KingSafety: 0.15,
        EarlyKingPenalty: 1,
        EarlyQueenPenalty: 1,
        PieceMobility: 0.8,
        CenterControl: 0.6,
        BishopPair: 0.05,
        OpenRookFile: 0.01,
        BishopAttacks: 0.05,
        DoubledPawns: 0.2,
        IsolatedPawns: 0.2,
        PieceInactivity: 0.7,
        WeakAttackers: 0,
    }

    # Weight map for middle game
    weight_map_mid = {
        Material: 1,
        PassedPawns: 0.01,
        KingSafety: 0.3,
        EarlyKingPenalty: 1,
        EarlyQueenPenalty: 1,
        PieceMobility: 0.8,
        CenterControl: 0.4,
        BishopPair: 0.1,
        OpenRookFile: 0.1,
        BishopAttacks: 0.25,
        DoubledPawns: 0.3,
        IsolatedPawns: 0.3,
        PieceInactivity: 0.4,
        WeakAttackers: 0,
    }

    # Weight map for endgame
    weight_map_end = {
        Material: 1,
        PassedPawns: 0.5,
        KingSafety: 0.1,
        PieceMobility: 0.6,
        EarlyKingPenalty: 0,
        EarlyQueenPenalty: -1,
        CenterControl: 0.2,
        BishopPair: 0.1,
        OpenRookFile: 0.2,
        BishopAttacks: 0.15,
        DoubledPawns: 0.2,
        IsolatedPawns: 0.2,
        PieceInactivity: 0.4,
        WeakAttackers: 0.2,
    }

    def __init__(self, *heuristics):
        """
            Arguments:
            :param heuristics: Object members of the heuristic class, used to cumulatively calculate the final position
            evaluation. Accepts n heuristic objects. If unprovided, the default is all defined heuristics in the system.
        """
        self.heuristics_use = []
        self.weights = None

        all_heuristics = Heuristic.__subclasses__()

        self.period = "Opening"

        if not heuristics:
            self.heuristics_use = list(map(lambda h: h(), all_heuristics))
        else:
            self.heuristics_use = [*heuristics]

        self.weightvec = self.__build_weight_vector(self.weight_map_open)

    @staticmethod
    def __game_phase(board: Board) -> str:
        """
            Determine the phase of the game according the current board state.

            Arguments:
            :param board: The current board state.
            :return: The phase of the game as a string, one of "Middlegame", "Opening" and "Endgame".
        """
        pieces = {
            KNIGHT: 3,
            BISHOP: 3,
            ROOK: 5,
            QUEEN: 9,
        }

        white_pawns = board.pieces(PAWN, WHITE)
        black_pawns = board.pieces(PAWN, BLACK)

        pieces_values_white, pieces_values_black = 0, 0

        for piece, value in zip(pieces.keys(), pieces.values()):
            pieces_values_white += len(list(board.pieces(piece, WHITE))) * value

        for piece, value in zip(pieces.keys(), pieces.values()):
            pieces_values_black += len(list(board.pieces(piece, BLACK))) * value

        if len(white_pawns) + len(black_pawns) >= 12:
            if pieces_values_white >= 28 and pieces_values_black >= 28:
                return "Opening"
            elif pieces_values_white >= 23 and pieces_values_black >= 23:
                return "Middlegame"
            else:
                return "Endgame"
        else:
            if pieces_values_white >= 23 and pieces_values_black >= 23:
                return "Middlegame"
            else:
                return "Endgame"

    def __build_weight_vector(self, with_map: Dict[PieceType, float]) -> np.array:
        """
            Helper function to build the weight vector. If n heuristic objects are passed, a weight vector of R^n is
            constructed according to the weight importances of the field weight_map.
            :return: A weight vector of R^n.
        """

        weight_vector = np.empty_like(self.heuristics_use)
        for i, heuristic in enumerate(self.heuristics_use):
            weight_vector[i] = with_map[type(heuristic)]

        # not needed unless using part of the heuristics (testing)
        # scaler = MinMaxScaler(feature_range=
        # (
        #     np.min(weight_vector),
        #     np.max(weight_vector)
        # )
        # )
        #
        # scaled = scaler.fit_transform(weight_vector.reshape(-1, 1)).flatten()
        # scaled = scaled / np.sum(scaled)

        return weight_vector

    def evaluate_position(self, board: Board) -> float:
        """
            Evaluation of any board position/state. A positive result is an advantage for white, the converse is an
            advantage for black. A win for white is given by np.inf, while -np.inf is a win for black.
            A zero evalution is a draw.

            Arguments:
            :param board: An arbitrary board state.
            :return: Inner product of the evaluation for white minus the inner product of the evaluation for black.
        """

        new_period = Evaluator.__game_phase(board)

        if new_period != self.period:
            self.period = new_period
            if new_period == "Middlegame":
                self.__build_weight_vector(self.weight_map_mid)
            elif new_period == "Opening":
                self.__build_weight_vector(self.weight_map_open)
            else:
                self.__build_weight_vector(self.weight_map_end)

        evaluation_vec_white = np.array([h.estimate(board, WHITE) for h in self.heuristics_use])
        evaluation_vec_black = np.array([h.estimate(board, BLACK) for h in self.heuristics_use])

        return np.dot(self.weightvec, evaluation_vec_white) \
            - np.dot(self.weightvec, evaluation_vec_black)
