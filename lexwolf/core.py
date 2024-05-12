from time import time

import chess
from random import shuffle, seed, randrange, choices, random
from lexwolf.bitBoard import bitBoard
import numpy as np


class LexWolfCore():
    """
    Parent class for all LexWolf chess models.
    """

    def __init__(self, max_depth=3, max_thinking_time=3600.0, random_seed=None, verbose=0, bruteForce = False,incrementalEvaluation=True):
        self.incrEval = incrementalEvaluation
        self.bF = bruteForce
        self.bitBrd = bitBoard(bruteForce=bruteForce)
        self.is_playing = False
        self.is_defeated = False
        self.is_winner = False
        self.combinations_count = 0
        self.max_depth = max_depth  # None = infinite
        self.verbose = verbose
        self.max_thinking_time = max_thinking_time  # Number of seconds before plays the move
        if random_seed is not None:
            seed(random_seed)
            np.random.seed(random_seed)
        if max_depth is None:
            self.max_depth = float('inf')

    def find_optimal_move(self, board=chess.Board()):
        pass

    def find_optimal_move_incremental(self, board=chess.Board()):
        pass

    def verbose_message(self, message):
        if self.verbose:
            print(message)

    def count_controlled_squares(self, board, color):
        """
        Counts the squares controlled by each piece type for the specified color.

        :param board: The chess board state.
        :param color: The color for which to count controlled squares (chess.WHITE or chess.BLACK).
        :return: Dictionary with the count of controlled squares for each piece type.
        """
        controlled_squares = {
            chess.PAWN: 0,
            chess.KNIGHT: 0,
            chess.BISHOP: 0,
            chess.ROOK: 0,
            chess.QUEEN: 0,
            chess.KING: 0
        }
        all_controlled_squares = set()

        for piece_type in controlled_squares.keys():
            for square in board.pieces(piece_type, color):
                legal_moves = board.legal_moves
                for move in legal_moves:
                    if move.from_square == square:
                        # Add to controlled squares if the move is legal (including captures)
                        all_controlled_squares.add(move.to_square)
                        # Special handling for pawns
                        if piece_type == chess.PAWN:
                            pawn_attacks = board.attacks(square)
                            all_controlled_squares.update(pawn_attacks)

        # Update the count for each piece type based on controlled squares
        for square in all_controlled_squares:
            attacked_piece = board.piece_at(square)
            if attacked_piece and attacked_piece.color == color:
                controlled_squares[attacked_piece.piece_type] += 1

        return controlled_squares


class DummyLexWolf(LexWolfCore):
    """
    Chooses a random legal move
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def find_optimal_move(self, board=chess.Board()):
        legal_moves = list(board.legal_moves)
        shuffle(legal_moves)
        return legal_moves[0]

    def find_optimal_move_incremental(self, board=chess.Board()):
        legal_moves = list(board.legal_moves)
        shuffle(legal_moves)
        return legal_moves[0]
