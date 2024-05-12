from time import time

import chess
from random import shuffle, seed, randrange, choices, random
import numpy as np
from lexwolf.core import LexWolfCore


class AdaptativeMinmaxLexWolf(LexWolfCore):
    def __init__(self, center_bonus=0.0, control_bonus=0.1, king_bonus=0.2, check_bonus=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_depth = self.max_depth
        self.center_bonus = center_bonus
        self.control_bonus = control_bonus
        self.king_bonus = king_bonus
        self.check_bonus = check_bonus
        self.start_time = time()
        self.incrEval = False

    def get_adaptative_depth(self, board):
        # Use `self.initial_depth` as the base depth
        pieces_left = len([piece for piece in board.piece_map().values() if piece])
        if pieces_left <= 4:
            return self.initial_depth + 5
        elif pieces_left <= 10:
            return self.initial_depth + 3
        elif pieces_left <= 16:
            return self.initial_depth + 1
        elif pieces_left <= 32:
            return self.initial_depth
        else:
            # Default or another logic for move counts outside the specified ranges
            return self.initial_depth

    def evaluate(self, board):
        # Standard piece values
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
        king_safety_squares = [chess.A1, chess.A8, chess.H1, chess.H8]

        # Initial score
        score = 0

        # Material and positional score
        for piece_type in piece_values:
            for square in board.pieces(piece_type, chess.WHITE):
                score += piece_values[piece_type]
                if square in center_squares:
                    score += self.center_bonus  # Central control bonus
            for square in board.pieces(piece_type, chess.BLACK):
                score -= piece_values[piece_type]
                if square in center_squares:
                    score -= self.center_bonus  # Central control bonus

        # King safety
        if self.king_bonus:
            for square in board.pieces(chess.KING, chess.WHITE):
                if square in king_safety_squares:
                    score += self.king_bonus  # King safety bonus
            for square in board.pieces(chess.KING, chess.BLACK):
                if square in king_safety_squares:
                    score -= self.king_bonus  # King safety bonus

        # Checkmate and stalemate
        if board.is_checkmate():
            if board.turn:
                score -= 1000  # Black wins
            else:
                score += 1000  # White wins
        elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            score = 0  # Draw

        # Add control bonus
        if self.control_bonus:
            white_control = self.count_controlled_squares(board, chess.WHITE)
            black_control = self.count_controlled_squares(board, chess.BLACK)
            score += self.control_bonus * sum(white_control.values())
            score -= self.control_bonus * sum(black_control.values())

        # Add check bonus
        if self.check_bonus and board.is_check():
            score += self.check_bonus

        return score

    def minimax(self, board, depth, is_maximizing):
        if depth == 0 or board.is_game_over():
            return self.evaluate(board)

        if is_maximizing:
            max_eval = float('-inf')
            for move in board.legal_moves:
                if time() - self.start_time > self.max_thinking_time:
                    break
                board.push(move)
                eval = self.minimax(board, depth - 1, False)
                board.pop()
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                if time() - self.start_time > self.max_thinking_time:
                    break
                board.push(move)
                eval = self.minimax(board, depth - 1, True)
                board.pop()
                min_eval = min(min_eval, eval)
            return min_eval

    def find_optimal_move(self, board=chess.Board()):
        self.start_time = time()

        self.get_adaptative_depth(board)

        legal_moves = list(board.legal_moves)
        shuffle(legal_moves)
        best_move = legal_moves[0]
        best_value = float('-inf') if board.turn == chess.WHITE else float('inf')

        for move in legal_moves:
            if time() - self.start_time > self.max_thinking_time:
                break
            board.push(move)
            self.combinations_count = 1
            board_value = self.minimax(board, self.max_depth - 1, not board.turn)
            board.pop()
            r = randrange(2)

            if board.turn == chess.WHITE:
                if board_value > best_value or (board_value == best_value and r == 0):
                    best_value = board_value
                    best_move = move
            else:
                if board_value < best_value or (board_value == best_value and r == 0):
                    best_value = board_value
                    best_move = move

        return best_move

    def safe_move(self, previous_move, new_move, board):
        if new_move in board.legal_moves:
            return new_move
        else:
            return previous_move