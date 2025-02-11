from chess import Board
from typing import Set, Tuple, Optional


class ExpectimaxNode(object):


    def __init__(self, board: Optional[Board] = Board(), parent_type: Optional[str] = "max"):
        """
        Constructor for an ExpectimaxNode.
        :param board: The board representing the current position.
        :param parent_type: Type of the parent node ("max" or "min" or "chance").
        """
        self.board = board
        self.children = set()  # Will hold the child nodes
        self.parent_type = parent_type  # Parent type (max, min, or chance)

    def calculate_children(self) -> Set[Tuple['ExpectimaxNode', float]]:
        """
        Generate all possible children nodes based on the legal moves.
        :return: A set of (ExpectimaxNode, probability) tuples.
        """
        legal_moves = list(self.board.legal_moves)
        total_moves = len(legal_moves)

        if total_moves == 0:
            return set()  # No legal moves (game over or stalemate)

        move_probabilities = []
        for move in legal_moves:
            child_board = self.board.copy()
            child_board.push(move)

            probability = 1.0 / total_moves
            child_node = ExpectimaxNode(child_board, "min" if self.parent_type == "max" else "max")
            move_probabilities.append((child_node, probability))

        return set(move_probabilities)

    def __str__(self) -> str:
        return f'{self.board.__str__()}'
