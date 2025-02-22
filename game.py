import numpy as np

import pandas as pd

import chess.pgn as pgn

from search.minimax import Minimax
from search.expectiminimax import Expectiminimax


from eval.evaluation import Evaluator

from chess import *

from colorama import Fore, init

def print_board_sf_style(b: Board):
    board_str = str(b)
    board_rows = board_str.split("\n")
    board_rows = [row.replace(" ", "") for row in board_rows]

    print("  +-----------------+")
    for i in range(8):
        row = board_rows[i]
        formatted_row = []
        for piece in row:
            if piece.isupper():  # White pieces
                formatted_row.append(Fore.WHITE + piece)
            elif piece.islower():  # Black pieces
                formatted_row.append(Fore.BLACK + piece)
            else:  # Empty squares
                formatted_row.append(Fore.RESET + '.')
        print(f"{8 - i} | {' '.join(formatted_row)}" + Fore.RESET + " "
                                                                    "|")
    print("  +-----------------+")
    print("    a b c d e f g h")

if __name__ == '__main__':

    evaluator = Evaluator()

    mini = Minimax(evaluator)
    expecti = Expectiminimax(evaluator)

    board = Board()

    answer = set(input("Who's " + Fore.RED + "Minimax" + Fore.RESET + "?").lower())

    mini_white = True

    if len(answer.intersection(set("black"))) == 5:
        print("Game started with " + Fore.RED + "Minimax" + Fore.RESET + " as black, " + Fore.BLUE + "Expectiminimax" +  Fore.RESET + " as white")
        mini_white = False
    elif len(answer.intersection(set("white"))) == 5:
        print("Game started with " + Fore.RED + "Minimax" + Fore.RESET + " as white, " + Fore.BLUE + "Expectiminimax" +  Fore.RESET + " as black")
    else:
        raise ValueError(f'Invalid input:\n{''.join(answer)}\nPlease enter either white or black')

    print_board_sf_style(board)

    game = pgn.Game()

    while not board.is_game_over():
        print(Fore.WHITE + "White (" + f'{ Fore.RED + 'Mini' if mini_white else  Fore.BLUE + 'expecti'}' + Fore.WHITE + ")" + Fore.RESET + "'s Turn")

        mini_eval, mini_move = mini.search(0, 4, mini_white, -np.inf, np.inf, board)
        board.push(mini_move)

        game = game.add_variation(mini_move)

        print_board_sf_style(board)

        if board.is_game_over():
            break

        print(Fore.BLACK + "Black (" + f'{ Fore.RED + 'mini' if not mini_white else  Fore.BLUE + 'Expecti'}' + Fore.BLACK + ")" + Fore.RESET + "'s Turn")

        expecti_eval, expecti_move = expecti.search(0, 3, not mini_white, board)
        board.push(expecti_move)

        game = game.add_variation(expecti_move)

        print_board_sf_style(board)
        break

    only_string = str(game.game()).split(']')[-1].strip()
    print(only_string)