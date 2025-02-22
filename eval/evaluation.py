from sklearn.preprocessing import MinMaxScaler
from eval.heuristics import (Heuristic, Material, PassedPawns, DoubledPawns, IsolatedPawns, BishopPair, BishopAttacks,
    CenterControl, KingSafety, EarlyKingPenalty, EarlyQueenPenalty, WeakAttackers, OpenRook, PieceMobility, PieceInactivity)
from chess import *

import numpy as np

class Evaluator:
    # === Static Attributes === #
    move_no = 1

    piece_presence = {
        BLACK: {
            BISHOP: 2,
            KNIGHT: 2,
            QUEEN: 1,
            ROOK: 1,
            PAWN: 8,
        },

        WHITE: {
            BISHOP: 2,
            KNIGHT: 2,
            QUEEN: 1,
            ROOK: 1,
            PAWN: 8,
        }
    }

    operation_stack = []

    weight_map_open = {
        # todo: make weights better :D :p
        Material: 0.8,
        PassedPawns: 0.01,
        KingSafety: 0.15,
        EarlyKingPenalty: 100,
        EarlyQueenPenalty: 50,
        PieceMobility: 0.5,
        CenterControl: 0.2,
        BishopPair: 0.05,
        OpenRook: 0.01,
        BishopAttacks: 0.01,
        DoubledPawns: 0.4,
        IsolatedPawns: 0.4,
        PieceInactivity: 0.4,
        WeakAttackers: 0.3,
        # Checkmate: 1
    }

    # todo: construct weight maps for middle and end games
    weight_map_mid = {
        Material: 0.50,
        PassedPawns: 0.15,
        KingSafety: 0.20,
        PieceMobility: 0.15,
        CenterControl: 0.1,
        BishopPair: 0.04,
        OpenRook: 0.05,
        BishopAttacks: 0.05,
        DoubledPawns: 0.03,
        IsolatedPawns: 0.05,
        PieceInactivity: 0.1,
        WeakAttackers: 0.2
    }

    weight_map_end = {
        Material: 0.50,
        PassedPawns: 0.3,
        KingSafety: 0.20,
        PieceMobility: 0.3,
        CenterControl: 0.05,
        BishopPair: 0.1,
        OpenRook: 0.1,
        BishopAttacks: 0.1,
        DoubledPawns: 0.2,
        IsolatedPawns: 0.2,
        PieceInactivity: 0.1,
        WeakAttackers: 0.2
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

        if not heuristics:
            self.heuristics_use = list(map(lambda h: h(), all_heuristics))
        else:
            self.heuristics_use = [*heuristics]

        self.weightvec = self.__build_weight_vector()

    def __build_weight_vector(self) -> np.array:
        """
            Helper function to build the weight vector. If n heuristic objects are passed, a weight vector of R^n is
            constructed according to the weight importances of the field weight_map.
            :return: A weight vector of R^n.
        """
        weight_vector = np.empty_like(self.heuristics_use)
        for i, heuristic in enumerate(self.heuristics_use):
            weight_vector[i] = Evaluator.weight_map_open[type(heuristic)]

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

        evaluation_vec_white = np.array([h.estimate(board, WHITE) for h in self.heuristics_use])
        evaluation_vec_black = np.array([h.estimate(board, BLACK) for h in self.heuristics_use])

        return np.dot(self.weightvec, evaluation_vec_white) \
             - np.dot(self.weightvec, evaluation_vec_black)


    @staticmethod
    def piececount_update(board: Board, move: Move) -> None:
        """
            Method that efficiently keeps track of material counts on the board.

            Arguments:
            :param board: An arbitrary board state.
            :param move: The move performed, that might affect the material count on the board state.
        """
        piece = board.piece_at(move.to_square)

        if piece is not None:
            Evaluator.piece_presence[piece.color][piece.piece_type] -= 1
            Evaluator.operation_stack.append((piece.color, piece.piece_type))

    @staticmethod
    def pop_upd_stack() -> None:
        """
            Undoes the last material count update issued by Evaluator.piececount_update().

            Arguments:
            :param board: An arbitrary board state.
            :param move: The move performed, that might affect the material count on the board state.
        """
        if len(Evaluator.operation_stack) != 0:
            undo = Evaluator.operation_stack.pop()
            Evaluator.piece_presence[undo[0]][undo[1]] += 1
