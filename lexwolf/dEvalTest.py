import chess
from bitBoard import bitBoard
from structEl import structEl
import numpy as np
import time
import cProfile
import pstats

board = chess.Board()
bboard = bitBoard(board)

if __name__ == "__main__":
    with cProfile.Profile() as profile:
        for move in list(board.legal_moves):
            board.push(move)
            bboard.setList(board)
            lastList = bboard.getList()
            lastVal = bboard.getEval()
            #print("first board, value: ", lastVal)
            #print(board)
            for move in list(board.legal_moves):
                board.push(move)
                bboard.setList(board)
                newVal = bboard.getEval()
                dVal = bboard.getDeltaEval(lastList, lastVal)
                #print("new board")
                #print(board)
                #print('newVal; ', newVal)
                #print("newVal - dVal: ",newVal - dVal)
                for move in list(board.legal_moves):
                    board.push(move)
                    bboard.setList(board)
                    newVal = bboard.getEval()
                    dVal = bboard.getDeltaEval(lastList, lastVal)
                    #print("new board")
                    #print(board)
                    #print('newVal; ', newVal)
                    #print("newVal - dVal: ",newVal - dVal)
                    board.pop()
                board.pop()
            board.pop()

profile.dump_stats("res4.prof")

results = pstats.Stats("res4.prof") # use "tuna res.prof" in CLI
results.sort_stats(pstats.SortKey.TIME)
#print(results)