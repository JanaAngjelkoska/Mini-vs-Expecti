import os
import sys

import numpy as np

import pandas as pd

import chess.pgn as pgn
from chess.pgn import GameNode

from search.minimax import Minimax
from search.expectiminimax import Expectiminimax

from eval.evaluation import Evaluator

from chess import *

from colorama import Fore

from pathlib import Path


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


def save_game(g: GameNode, result: str) -> None:
    only_string = str(g.game()).split(']')[-1].strip()

    to_save = {
        'pgn': [only_string],
        'result': [result]
    }

    namedir = 'games'
    played_dir = Path(namedir)
    played_dir.mkdir(exist_ok=True)

    if not any(played_dir.iterdir()):
        df = pd.DataFrame(to_save)
    else:
        df = pd.read_csv(f'./{namedir}/played.csv')
        if ((df['pgn'] == only_string) & (df['result'] == result)).any():
            print(f"Variation:\n{only_string}\nalready exists in the database.", file=sys.stderr)
            return
        df_concat = pd.DataFrame(to_save)
        print(df_concat)
        df = pd.concat([df, df_concat], axis=0, ignore_index=True)

    print('new df:')
    print(df)

    df.to_csv(f'./{namedir}/played.csv', index=False)


if __name__ == '__main__':

    evaluator = Evaluator()

    mini = Minimax(evaluator)
    expecti = Expectiminimax(evaluator)

    board = Board(fen=STARTING_FEN)

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

    game = pgn.Game()
    game.setup(board)

    result = None

    while not board.is_game_over():
        print(
            Fore.WHITE + "White (" + f'{Fore.RED + 'Mini' if mini_white else Fore.BLUE + 'expecti'}' + Fore.WHITE + ")" + Fore.RESET + "'s Turn")
        
        white_move = None
        
        if mini_white:
            mini_eval, mini_move = mini.search(0, 4, WHITE, -np.inf, np.inf, board)
            white_move = mini_move
        else:
            expecti_eval, expecti_move = expecti.search(0, 3, WHITE, board)
            white_move = expecti_move

        print("White moved: ", board.san(white_move))
        game = game.add_variation(white_move)

        board.push(white_move)

        if board.is_game_over():
            if board.is_checkmate():
                result = '1-0'
                print('Checkmate, White is the winner.')
            else:
                result = '1/2-1/2'
                print('Draw.')
                
            break

        print_board_sf_style(board)

        print(
            Fore.BLACK + "Black (" + f'{Fore.RED + 'mini' if not mini_white else Fore.BLUE + 'Expecti'}' + Fore.BLACK + ")" + Fore.RESET + "'s Turn")

        black_move = None
        
        if mini_white:
            expecti_eval, expecti_move = expecti.search(0, 3, BLACK, board)
            black_move = expecti_move
        else:
            mini_eval, mini_move = mini.search(0, 4, BLACK, -np.inf, np.inf, board)
            black_move = mini_move

        print("Black moved: ", board.san(black_move))
        game = game.add_variation(black_move)

        board.push(black_move)

        if board.is_game_over():
            if board.is_checkmate():
                result = '0-1'
                print('Checkmate, Black is the winner.')
            else:
                result = '1/2-1/2'
                print('Draw')
            
            break

        print_board_sf_style(board)

    
    print_board_sf_style(board)
    save_game(game, result)
