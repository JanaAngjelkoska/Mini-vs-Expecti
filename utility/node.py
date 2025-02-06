from chess import Board, LegalMoveGenerator

from typing import Set, Optional

from utility.node_exceptions import NoChildrenCalculatedYetException

class PositionNode(object):
    """
        A single unit, or state (a.k.a. node) in a chess search tree. This class is only illustrative, to demonstrate
        how chess positions and combinations after each move work.
    """
    def __init__(self, root: Optional[Board] = Board()):
        """
        Constructor for a PositionNode.
            :param root: A predefined root checkpoint (position) to start building upon. By default, this is the
            starting position in a chess board defined by the FEN:
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        """
        self.board = Board(root.fen())
        self.children = set()
        self.generator = LegalMoveGenerator(self.board)

    def next_moves(self):
        return self.generator

    def calculate_children(self) -> Set['PositionNode']:
        """
        Calculates all the next states (children positions) of the current position.
            :return: All possible next positions, each wrapped in a PositionNode.
        """

        for move in self.generator:
            child_board = Board(self.board.fen())
            child_board.push(move)
            self.children.add(PositionNode(child_board))

        return self.children

    def get_children(self) -> Set['PositionNode']:
        if len(self.children) == 0:
            raise NoChildrenCalculatedYetException(self.__str__())
        return self.children

    @staticmethod
    def build_tree(depth, root: Optional[Board] = Board()) -> 'PositionNode':
        """
        ONLY TO BE USED FOR EXPERIMENTAL PURPOSES
        Constructs a tree of all possible positions after depth :param depth.
            :param depth: Maximum depth of the tree being built.
            :param root: A starting position.
            :return: The root of the built tree.
        """
        this_node = PositionNode(root)
        PositionNode.__helper_build(0, depth, this_node, this_node)
        return this_node

    @staticmethod
    def __helper_build(current_depth: int, max_depth: int, root: 'PositionNode', original: 'PositionNode') -> None:
        """
        Helper private recursive method for a building a tree of positions.
            :param current_depth: Current depth, typically 0 is first sent.
            :param max_depth: Maximum depth of the tree being built.
            :param root: The current node being built upon.
            :param original: The original root node to be returned.
            :return: void
        """
        if max_depth == current_depth:
            return

        lookup_children = root.calculate_children()

        for child in lookup_children:
            PositionNode.__helper_build(current_depth + 1, max_depth, child, original)

    def __str__(self) -> str:
        presuffix = '- - - - - - - -'
        return f'{presuffix}\n{self.board.__str__()}\n{presuffix}'
