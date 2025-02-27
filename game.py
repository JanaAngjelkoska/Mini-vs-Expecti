import os

import numpy as np

import pandas as pd

import chess.pgn as pgn
from chess.pgn import GameNode

from search.minimax import Minimax
from search.expectiminimax import Expectiminimax

from eval.evaluation import Evaluator

from chess import *

from colorama import Fore


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


def save_game(g: GameNode) -> None:
    only_string = str(g.game()).split(']')[-1].strip()
    result = only_string.split(' ')[-1]

    if result == '*':
        raise ValueError(f"Cannot save game with PGN: {only_string}\nGame is yet to finish")  # game in progress
    elif result == '1-0':
        result = '1'  # white victory
    elif result == '0-1':
        result = '0'  # black victory
    else:
        result = '2'  # draw

    to_save = {
        'pgn': only_string,
        'result': result
    }

    if len(os.listdir('./games')) == 0:
        df = pd.DataFrame(to_save)
    else:
        df = pd.read_csv('./games/played.csv')
        df.append(to_save)

    df.to_csv('./games/played.csv')


if __name__ == '__main__':

    game = pgn.Game()

    automated = input("Would you like to automate the first move in all games?").lower()
    first_moves = ['b1c3', 'g1f3', 'c2c4', 'e2e4', 'd2d4']
    has_played_first = False

    for first_move in first_moves:

        evaluator = Evaluator()

        mini = Minimax(evaluator)
        expecti = Expectiminimax(evaluator)

        board = Board()

        answer = set(input("Who's " + Fore.RED + "Minimax" + Fore.RESET + "?").lower())

        mini_white = True

        if len(answer.intersection(set("black"))) == 5:
            print(
                "Game started with " + Fore.RED + "Minimax" + Fore.RESET + " as black, " + Fore.BLUE + "Expectiminimax" + Fore.RESET + " as white")
            mini_white = False
        elif len(answer.intersection(set("white"))) == 5:
            print(
                "Game started with " + Fore.RED + "Minimax" + Fore.RESET + " as white, " + Fore.BLUE + "Expectiminimax" + Fore.RESET + " as black")
        else:
            raise ValueError(f'Invalid input:\n{''.join(answer)}\nPlease enter either white or black')

        print_board_sf_style(board)

        while not board.is_game_over():
            print(
                Fore.WHITE + "White (" + f'{Fore.RED + 'Mini' if mini_white else Fore.BLUE + 'expecti'}' + Fore.WHITE + ")" + Fore.RESET + "'s Turn")
            white_move = None

            if automated and not has_played_first:
                print("Automated move: ", first_move)
                white_move = first_move
                board.push_san(first_move)
                has_played_first = True
            else:
                if mini_white:
                    mini_eval, mini_move = mini.search(0, 4, mini_white, -np.inf, np.inf, board)
                    white_move = mini_move
                else:
                    expecti_eval, expecti_move = expecti.search(0, 3, not mini_white, board)
                    white_move = expecti_move

                board.push(white_move)
                game = game.add_variation(white_move)

            print("White moved: ", white_move)

            print_board_sf_style(board)

            if board.is_game_over():
                has_played_first = not has_played_first
                break

            print(
                Fore.BLACK + "Black (" + f'{Fore.RED + 'mini' if not mini_white else Fore.BLUE + 'Expecti'}' + Fore.BLACK + ")" + Fore.RESET + "'s Turn")

            black_move = None
            if mini_white:
                expecti_eval, expecti_move = expecti.search(0, 3, not mini_white, board)
                black_move = expecti_move
            else:
                mini_eval, mini_move = mini.search(0, 4, mini_white, -np.inf, np.inf, board)
                black_move = mini_move

            board.push(black_move)
            print("Black moved: ", black_move)

            game = game.add_variation(black_move)

            print_board_sf_style(board)
            print(Evaluator.piece_presence)

        save_game(game)
        has_played_first = False
