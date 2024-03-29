from time import time

import chess
from random import shuffle, randrange
from lexwolf.core import LexWolfCore
from lexwolf.bitBoard import bitBoard


class MinmaxLexWolf(LexWolfCore):
    def __init__(self, center_bonus=0.1, control_bonus=0.1, king_bonus=0.2, check_bonus=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.center_bonus = center_bonus
        self.control_bonus = control_bonus
        self.king_bonus = king_bonus
        self.check_bonus = check_bonus
        self.start_time = time()

    def checkEqual(self, board, res):
        exhaustive = self.evaluate(board)
        if (abs(res - exhaustive) >= 1e-10 and abs(res - exhaustive) != 100000):
            lastboard = board.copy()
            lastboard.pop()
            self.bitBrd.setList(lastboard)
            print("last board, value: ", self.bitBrd.getEval())
            print(lastboard)
            print(lastboard.fen())
            print("next board, values: (exh - incr)", exhaustive, ' - ', res)
            print(board)
            print(board.fen())
            print("abs(res-exhaustive): ", abs(res - exhaustive))
            raise ValueError("The results of the incremental evaluation and the exhaustive evaluation diverge")

    def evaluate(self, board):

        # Initial score
        score = 0

        # Material and positional score
        self.bitBrd.setList(board)
        score = self.bitBrd.getEval()

        # Checkmate and stalemate
        if board.is_checkmate():
            if board.turn:
                score -= 100000  # Black wins
            else:
                score += 100000  # White wins
        elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            score = 0  # Draw

        # Add control bonus
        # if self.control_bonus:
        #    white_control = self.count_controlled_squares(board, chess.WHITE)
        #    black_control = self.count_controlled_squares(board, chess.BLACK)
        #    score += self.control_bonus * sum(white_control.values())
        #    score -= self.control_bonus * sum(black_control.values())

        # Add check bonus
        # if self.check_bonus and board.is_check():
        # score += self.check_bonus

        return score

    def evaluate_incremental(self, staticVal, board):

        # Initial score
        score = staticVal

        # Checkmate and stalemate
        if board.is_checkmate():
            if board.turn:
                score -= 100000  # Black wins
            else:
                score += 100000  # White wins
        elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            score = 0  # Draw

        # Add control bonus
        # if self.control_bonus:
        #    white_control = self.count_controlled_squares(board, chess.WHITE)
        #    black_control = self.count_controlled_squares(board, chess.BLACK)
        #    score += self.control_bonus * sum(white_control.values())
        #    score -= self.control_bonus * sum(black_control.values())

        # Add check bonus
        # if self.check_bonus and board.is_check():
        #    score += self.check_bonus

        return score

    def minimax(self, board, depth, alpha, beta, is_maximizing):
        if depth == 0 or board.is_game_over():
            return self.evaluate(board)

        if is_maximizing:
            max_eval = float('-inf')
            for move in board.legal_moves:
                if time() - self.start_time > self.max_thinking_time:
                    break
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, max_eval)
                if alpha >= beta:
                    break  # Beta cut-off
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                if time() - self.start_time > self.max_thinking_time:
                    break
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, min_eval)
                if beta <= alpha:
                    break  # Alpha cut-off
            return min_eval

    def minimax_incremental(self, board, boardStaticVal, prevList, depth, alpha, beta, is_maximizing):
        if depth == 0 or board.is_game_over():
            res = self.evaluate_incremental(boardStaticVal, board)
            # self.checkEqual(board,res)
            return res

        if is_maximizing:
            max_eval = float('-inf')
            for move in board.legal_moves:
                if time() - self.start_time > self.max_thinking_time:
                    break
                board.push(move)
                self.bitBrd.setList(board)
                staticVal = self.bitBrd.getDeltaEval(prevList, boardStaticVal)
                lastList = self.bitBrd.getList()
                # self.checkEqual(board,staticVal)
                eval = self.minimax_incremental(board, staticVal, lastList, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, max_eval)
                if alpha >= beta:
                    break  # Beta cut-off
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                if time() - self.start_time > self.max_thinking_time:
                    break
                board.push(move)
                self.bitBrd.setList(board)
                staticVal = self.bitBrd.getDeltaEval(prevList, boardStaticVal)
                lastList = self.bitBrd.getList()
                # self.checkEqual(board,staticVal)
                eval = self.minimax_incremental(board, staticVal, lastList, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, min_eval)
                if beta <= alpha:
                    break  # Alpha cut-off
            return min_eval

    def quicksort(self, moveList, turn):
        if (len(moveList) <= 1):
            return moveList
        elif (turn == chess.WHITE):
            pivot = moveList[0][1]
            less_than_pivot = [x for x in moveList[1:] if x[1] <= pivot]
            greater_than_pivot = [x for x in moveList[1:] if x[1] > pivot]
            return self.quicksort(greater_than_pivot, turn) + [moveList[0]] + self.quicksort(less_than_pivot, turn)
        elif (turn == chess.BLACK):
            pivot = moveList[0][1]
            less_than_pivot = [x for x in moveList[1:] if
                               x[1] > pivot]  # for black, lesser moves are those of a greater value
            greater_than_pivot = [x for x in moveList[1:] if x[1] <= pivot]
            return self.quicksort(greater_than_pivot, turn) + [moveList[0]] + self.quicksort(less_than_pivot, turn)

    def find_optimal_move(self, board=chess.Board()):
        turn = board.turn
        self.start_time = time()
        legal_moves = list(board.legal_moves)
        move_value = [None] * len(legal_moves)

        for i in range(len(legal_moves)):
            board.push(legal_moves[i])
            move_value[i] = (legal_moves[i], self.minimax(board, 0, float('-Inf'), float('Inf'), not turn))
            board.pop()

        sorted_tuples = self.quicksort(move_value, turn)
        sorted_moves = [sorted_tuples[i][0] for i in range(len(legal_moves))]
        best_move = sorted_moves[0]
        best_value = float('-inf') if turn == chess.WHITE else float('inf')
        alpha = float('-inf')
        beta = float('inf')

        for move in sorted_moves:
            if time() - self.start_time > self.max_thinking_time:
                break
            board.push(move)
            self.combinations_count = 1
            board_value = self.minimax(board, self.max_depth - 1, alpha, beta, not turn)
            board.pop()
            r = randrange(2)

            if board.turn == chess.WHITE:
                if board_value > best_value or (board_value == best_value and r == 0):
                    best_value = board_value
                    best_move = move
                    alpha = max(alpha, best_value)  # Update alpha
            else:
                if board_value < best_value or (board_value == best_value and r == 0):
                    best_value = board_value
                    best_move = move
                    beta = min(beta, best_value)  # Update beta

        return best_move

    def find_optimal_move_incremental(self, board=chess.Board()):
        turn = board.turn
        self.start_time = time()
        legal_moves = list(board.legal_moves)
        move_value = [None] * len(legal_moves)

        for i in range(len(legal_moves)):
            board.push(legal_moves[i])
            move_value[i] = (legal_moves[i], self.minimax(board, 0, float('-Inf'), float('Inf'), not turn))
            board.pop()

        sorted_tuples = self.quicksort(move_value, turn)
        sorted_moves = [sorted_tuples[i][0] for i in range(len(legal_moves))]
        best_move = sorted_moves[0]
        best_value = float('-inf') if turn == chess.WHITE else float('inf')
        alpha = float('-inf')
        beta = float('inf')

        self.bitBrd.setList(board)
        prevList = self.bitBrd.getList()
        boardStaticVal = self.bitBrd.getEval()

        for move in sorted_moves:
            if time() - self.start_time > self.max_thinking_time:
                break
            board.push(move)
            self.bitBrd.setList(board)
            staticVal = self.bitBrd.getDeltaEval(prevList, boardStaticVal)
            if (randint(0, 1000) == 500):  # check if no deltaEval ~ getEval discrepancy at regular intervalls
                self.checkEqual(board, staticVal)
            lastList = self.bitBrd.getList()
            self.combinations_count = 1
            board_value = self.minimax_incremental(board, staticVal, lastList, self.max_depth - 1, alpha, beta,
                                                   not turn)
            board.pop()
            r = randrange(2)

            if board.turn == chess.WHITE:
                if board_value > best_value or (board_value == best_value and r == 0):
                    best_value = board_value
                    best_move = move
                    alpha = max(alpha, best_value)  # Update alpha
            else:
                if board_value < best_value or (board_value == best_value and r == 0):
                    best_value = board_value
                    best_move = move
                    beta = min(beta, best_value)  # Update beta

        return best_move

    def safe_move(self, previous_move, new_move, board):
        if new_move in board.legal_moves:
            return new_move
        else:
            return previous_move
