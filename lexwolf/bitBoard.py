import chess
import numpy as np

class bitBoard:

    def __init__(self,board=None):
        self.__list = np.asarray([None]*(768+5)) # private 772-bits bit array: 64*6*2 + 5 ~ 64 cases for each piece of each color plus side to move & castling rights

        if(board==None): # no board provided, initial board is supposed
            temp = chess.Board()
            self.setList(temp)
        elif(board!=None):
            self.setList(board)

    def getList(self):
        return self.__list

    def setList(self,board):
            # Dictionary to map piece colors to their index for bitboard representation
            COLOR_OFFSET = {chess.WHITE: 0, chess.BLACK: 1}
            BOOL_BIN = {True: 1, False: 0}
            self.__list = np.asarray([0b0]*(768+5))
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if(piece!=None):
                    index = square + (piece.piece_type-1)*64 + COLOR_OFFSET[piece.color]*64*6
                    self.__list[index] = 0b1

            self.__list[-5] = BOOL_BIN[board.turn]
            self.__list[-4] = BOOL_BIN[board.has_kingside_castling_rights(chess.WHITE)]
            self.__list[-3] = BOOL_BIN[board.has_queenside_castling_rights(chess.WHITE)]
            self.__list[-2] = BOOL_BIN[board.has_kingside_castling_rights(chess.BLACK)]
            self.__list[-1] = BOOL_BIN[board.has_queenside_castling_rights(chess.BLACK)]


