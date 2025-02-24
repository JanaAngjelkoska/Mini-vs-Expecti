import matplotlib.pyplot as plt
import networkx as nx
import customtkinter


class GUI:
    def __init__(self, board):
        self.board = board

    def create_canvas(self, white, black):
        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme('dark-blue')

        root = customtkinter.CTk()
        root.geometry("800x600")


        # this is the main frame where the board and the opponents names are shown
        left_main_frame = customtkinter.CTkFrame(master=root, text=black)
        left_main_frame.pack(padx=10, pady=10)

        black_player = customtkinter.CTkFrame(master=left_main_frame, text=black)
        black_player.pack(padx=10, pady=10, expand=True, fill="both")

        board_frame = customtkinter.CTkFrame(master=left_main_frame)
        board_frame.pack(padx=10, pady=10, expand=True, fill="both")

        white_player = customtkinter.CTkFrame(master=left_main_frame, text=white)
        white_player.pack(padx=10, pady=10, expand=True, fill="both")

        # TODO: the right side of the UI where the game is displayed in notation
        root.mainloop()