import chess
import numpy as np
from weights import WEIGHTS
from lexwolf.structEl import structEl
from math import floor

COLOR_OFFSET = {chess.WHITE: 0, chess.BLACK: 1}
BOOL_BIN = {True: 1, False: 0}

a1 = 100  # material
a2 = 2  # b ctl
a3 = 2  # q ctl
a4 = 2  # r ctl
a5 = 2  # k ctl
a6 = 2  # p ctl
a7 = 2  # p pen
a8 = 2  # king exposed
a9 = 1  # king protected


class bitBoard:

    def __init__(self, board=None, bruteForce=False):
        self.bF = bruteForce
        self.__list = np.asarray([None] * (
                    768 + 5))  # private 773-bits bit array: 64*6*2 + 5 ~ 64 cases for each piece of each color plus side to move & castling rights
        self.__pieceswght = np.asarray([None] * (768 + 5))
        self.setWeights()
        self.strEl = structEl()
        if (board == None):  # no board provided, initial board is supposed
            self.board = chess.Board()
        else:
            self.board = board

        self.setList(self.board)
        self.__value = 0

    def value(self):
        return self.__value

    def getList(self):
        return self.__list

    def getWeighted(self, lst=None):
        if lst is None:
            return self.__list * self.__pieceswght
        else:
            return lst * self.__pieceswght

    def getEval(self):
        if (self.bF == True):
            res = a1 * (np.sum(self.getWeighted()[0:6 * 64]) - np.sum(self.getWeighted()[
                                                                      6 * 64:12 * 64])) + a2 * self.bishopCtlBf() + a3 * self.queenCtlBf() + a4 * self.rookCtlBf() + a5 * self.knightCtl() + a6 * self.pawnCtl() - a7 * self.pawnPenalty() + a9 * self.kingProtected()
        else:
            # print("res exhaustive")
            res = a1 * (np.sum(self.getWeighted()[0:6 * 64]) - np.sum(self.getWeighted()[6 * 64:12 * 64]))
            # print("res material balance: ",res)
            res += a2 * self.bishopCtl()
            # print("res bctl: ",res)
            res += a3 * self.queenCtl()
            # print("res qctl: ",res)
            res += a4 * self.rookCtl()
            # print("res rkctl: ",res)
            res += a5 * self.knightCtl()
            # print("res knctl: ",res)
            res += a6 * self.pawnCtl()
            # print("res pwctl: ",res)
            res -= a7 * self.pawnPenalty()
            # print("res pwpen: ",res)
            res += a9 * self.kingProtected()
            # print("res kgprotected: ",res)

        self.__value = res
        return res

    def getDeltaEval(self, prevList, prevBboardValue):
        prevBboardList = prevList.copy()
        listDiff = self.__list - prevBboardList
        nbMoving = sum(abs(listDiff))
        superimposedList = np.ones(64)
        rankKing = []
        Castle = False
        colorMoved = None
        indexes = []
        pair = []
        for j in range(12):
            for µ in range(64):
                i = 64 * j + µ
                if listDiff[i] != 0:
                    superimposedList = superimposedList * listDiff[64 * j:64 * (j + 1)]
                    if j in [5, 11]:
                        rankKing.append(i)
                        if (len(rankKing) == 2 and abs(rankKing[0] - rankKing[1]) == 2):
                            Castle = True
                    if indexes:
                        if (i - indexes[-1] >= 64):
                            indexes.append(i)
                        else:
                            # print('pop indexes ',indexes[-1] ," ",i)
                            prevBboardList[indexes[
                                -1]] = 0  # removing the moving pieces from the previous board to facilitate computations
                            prevBboardList[i] = 0
                            pair.extend([i, indexes[-1]])
                            colorMoved = True if i < 6 * 64 else False
                    if not indexes:
                        indexes.append(i)

        for i in indexes:  # putting the vanishing piece's index as the very last position of the index list
            if i not in pair:
                indexes.remove(i)
                indexes.append(i)

        rankKing = np.asarray(rankKing)
        if (len(indexes) <= 1):
            superimposedList = np.zeros(64)

        weightedDiff = self.getWeighted(listDiff)
        res = a1 * (np.sum(weightedDiff[0:6 * 64]) - np.sum(weightedDiff[6 * 64:12 * 64]))
        # print("res material balance: ", res)
        # res += a9*(self.kingProtected(self.__list.copy())-self.kingProtected(prevBboardList))

        res = self.deltaIntrinsic(res, listDiff, prevBboardList, superimposedList, indexes, rankKing,
                                  Castle)  # intrinsic cost of moving the pieces related to their own modified range of influence

        res = self.deltaExtrinsic(res, listDiff, prevBboardList, weightedDiff, superimposedList, rankKing, Castle,
                                  colorMoved)  # extrinsic cost of moving the pieces related to the modified range of influence of all other pieces

        self.__value = res + prevBboardValue

        return self.__value

    def deltaIntrinsic(self, res, listDiff, prevList, superimposedList, indexes, rankKing, Castle):
        # print("Intrinsic")

        # print('indexes', indexes)

        prevBboardList = prevList.copy()
        for i in indexes:

            if (i / 64 >= 6):
                color = False

            if (i / 64 < 6):
                color = True

            if (superimposedList[i % 64] == -1 and listDiff[i] == -1):  # if the piece is eaten
                for j in indexes:
                    if (j != i):  # find the eating piece
                        line = floor(j / 64)
                        prevBboardList[line * 64:(line + 1) * 64] = prevBboardList[line * 64:(line + 1) * 64] + abs(
                            listDiff[line * 64:(
                                                           line + 1) * 64])  # accounting for loss of influence of piece eaten over moving piece

            if (i in np.asarray(range(0, 64))):  # white pawn moved
                ctl = a6 * self.subPawnCtl(color, prevBboardList, listDiff[0:64])
                # print('whit pawn ctl: ', ctl)
                pen = - a7 * self.subPawnPenaltyDel(color, prevBboardList, listDiff[0:64])
                # print('whit pawn ctl: ', ctl)
                res += ctl + pen


            elif (i in np.asarray(range(64, 2 * 64))):  # white knight moved
                noctld = self.subKnightCtl(color, prevBboardList, listDiff[64:2 * 64])
                ctl = a5 * noctld
                res += ctl


            elif (i in np.asarray(range(2 * 64, 3 * 64))):  # white bishop moved
                ctl = a2 * self.subBishopCtl(color, prevBboardList, listDiff[2 * 64:3 * 64])
                res += ctl


            elif (i in np.asarray(range(3 * 64, 4 * 64))):  # white rook moved
                if (Castle == False):
                    ctl = a4 * self.subRookCtl(color, prevBboardList, listDiff[3 * 64:4 * 64])
                else:
                    listDiffpos = (listDiff[3 * 64:4 * 64] + abs(listDiff[3 * 64:4 * 64])) / 2
                    listDiffneg = (listDiff[3 * 64:4 * 64] - abs(listDiff[3 * 64:4 * 64])) / 2
                    prevBboardListpos = prevBboardList.copy()
                    prevBboardListneg = prevBboardList.copy()
                    for j in rankKing:
                        prevBboardListpos[j] = 0
                        prevBboardListneg[j] = 1
                    ctl = a4 * (self.subRookCtl(color, prevBboardListpos, listDiffpos) + self.subRookCtl(color,
                                                                                                         prevBboardListneg,
                                                                                                         listDiffneg))

                res += ctl

            elif (i in np.asarray(range(4 * 64, 5 * 64))):  # white queen moved
                # print('listDiff[4*64:5*64]: ', listDiff[4*64:5*64])
                ctl = a3 * self.subQueenCtl(color, prevBboardList, listDiff[4 * 64:5 * 64])
                # print("sponge prev: ", self.sponge(prevBboardList))
                # print("white queen ctl: ", ctl)
                res += ctl


            elif (i in np.asarray(range(5 * 64, 6 * 64))):  # white king moved
                if (Castle == False):
                    ctl = a9 * self.subKingProtected(color, prevBboardList, listDiff[5 * 64:6 * 64])
                else:
                    listDiffpos = (listDiff[5 * 64:6 * 64] + abs(listDiff[5 * 64:6 * 64])) / 2
                    listDiffneg = (listDiff[5 * 64:6 * 64] - abs(listDiff[5 * 64:6 * 64])) / 2
                    prevBboardListpos = prevBboardList.copy()
                    prevBboardListpos[3 * 64:4 * 64] = (listDiff[3 * 64:4 * 64] + abs(listDiff[3 * 64:4 * 64])) / 2
                    ctl = a9 * (self.subKingProtected(color, prevBboardListpos, listDiffpos) + self.subKingProtected(
                        color, prevBboardList, listDiffneg))

                res += ctl

            elif (i in np.asarray(range(6 * 64, 7 * 64))):  # black pawn moved
                ctl = a6 * self.subPawnCtl(color, prevBboardList, listDiff[6 * 64:7 * 64])
                pen = - a7 * self.subPawnPenaltyDel(color, prevBboardList, listDiff[6 * 64:7 * 64])
                # print("black pawn ctl: ", ctl)
                # print('black pawn penalty: ', pen)
                res -= ctl + pen

            elif (i in np.asarray(range(7 * 64, 8 * 64))):  # black knight moved
                ctl = a5 * self.subKnightCtl(color, prevBboardList, listDiff[7 * 64:8 * 64])
                res -= ctl


            elif (i in np.asarray(range(8 * 64, 9 * 64))):  # black bishop moved
                ctl = a2 * self.subBishopCtl(color, prevBboardList, listDiff[8 * 64:9 * 64])
                res -= ctl

            elif (i in np.asarray(range(9 * 64, 10 * 64))):  # black rook moved
                if (Castle == False):
                    ctl = a4 * self.subRookCtl(color, prevBboardList, listDiff[9 * 64:10 * 64])
                else:
                    listDiffpos = (listDiff[9 * 64:10 * 64] + abs(listDiff[9 * 64:10 * 64])) / 2
                    listDiffneg = (listDiff[9 * 64:10 * 64] - abs(listDiff[9 * 64:10 * 64])) / 2
                    prevBboardListneg = prevBboardList.copy()
                    for j in rankKing:
                        prevBboardListneg[j] = 1
                    ctl = a4 * (self.subRookCtl(color, prevBboardList, listDiffpos) + self.subRookCtl(color,
                                                                                                      prevBboardListneg,
                                                                                                      listDiffneg))
                res -= ctl

            elif (i in np.asarray(range(10 * 64, 11 * 64))):  # black queen moved
                ctl = a3 * self.subQueenCtl(color, prevBboardList, listDiff[10 * 64:11 * 64])
                res -= ctl

            elif (i in np.asarray(range(11 * 64, 12 * 64))):  # black king moved
                if (Castle == False):
                    ctl = a9 * self.subKingProtected(color, prevBboardList, listDiff[11 * 64:12 * 64])
                else:
                    listDiffpos = (listDiff[11 * 64:12 * 64] + abs(listDiff[11 * 64:12 * 64])) / 2
                    listDiffneg = (listDiff[11 * 64:12 * 64] - abs(listDiff[11 * 64:12 * 64])) / 2
                    prevBboardListpos = prevBboardList.copy()
                    prevBboardListpos[9 * 64:10 * 64] = (listDiff[9 * 64:10 * 64] + abs(listDiff[9 * 64:10 * 64])) / 2
                    ctl = a9 * (self.subKingProtected(color, prevBboardListpos, listDiffpos) + self.subKingProtected(
                        color, prevBboardList, listDiffneg))
                res -= ctl

                # print("res intrinsic: ", res)
        return res

    def deltaExtrinsic(self, res, listDiff, prevBboardList, weightedDiff, superimposedList, rankKing, Castle,
                       colorMoved):
        # print("extrinsic")

        wdiffSponge = self.sponge(weightedDiff)
        prevSponge = self.sponge(prevBboardList)
        dirb1 = [self.strEl.NE, self.strEl.SW]
        dirb2 = [self.strEl.NW, self.strEl.SE]
        dirbis = dirb1 + dirb2
        dirk1 = [self.strEl.S, self.strEl.N]
        dirk2 = [self.strEl.E, self.strEl.W]
        dirrk = dirk1 + dirk2

        for k in range(len(wdiffSponge)):
            pivot = wdiffSponge[k]
            equiXchange = True if (superimposedList[k] != 0 and pivot == 0) else False
            King = True if k in rankKing % 64 else False
            evanescentKing = False
            kingColor = False
            if (pivot != 0 or King == True or equiXchange == True):
                ## #print("King == ", King, 'k: ', k ,'rankKing: ', rankKing)
                if (King == True):
                    kingColor = False if (rankKing[0] >= 6 * 64) else True
                    if (listDiff[5 * 64 + k] < 0 or listDiff[11 * 64 + k] < 0):
                        evanescentKing = True
                    elif (listDiff[5 * 64 + k] > 0 or listDiff[11 * 64 + k] > 0):
                        evanescentKing = False

                if (King == False and Castle == True):
                    for i in rankKing:
                        if (listDiff[i] == -1):
                            prevBboardList[i] = 1 if pivot < 0 else 0
                    prevSponge = self.sponge(prevBboardList)

                convkg = self.strEl.Kg[k]
                convkn = self.strEl.Kn[k]
                convbp = self.strEl.wP[k]
                convwp = self.strEl.bP[k]
                convbis = np.asarray([0] * 64)
                convrk = np.asarray([0] * 64)
                convqn = np.asarray([0] * 64)
                for d in dirbis:
                    j = 0
                    while (d[-j][k] @ prevSponge > 1):
                        j += 1
                    convbis += d[-j][k]
                for d in dirrk:
                    j = 0
                    while (d[-j][k] @ prevSponge > 1):
                        j += 1
                    convrk += d[-j][k]
                convqn = convbis + convrk

                nwkg = convkg @ prevBboardList[5 * 64:6 * 64]
                nbkg = convkg @ prevBboardList[11 * 64:12 * 64]
                nwkn = convkn @ prevBboardList[1 * 64:2 * 64]
                nbkn = convkn @ prevBboardList[7 * 64:8 * 64]
                nwbs = convbis @ prevBboardList[2 * 64:3 * 64]
                nbbs = convbis @ prevBboardList[8 * 64:9 * 64]
                nwrk = convrk @ prevBboardList[3 * 64:4 * 64]
                nbrk = convrk @ prevBboardList[9 * 64:10 * 64]
                nwq = convqn @ prevBboardList[4 * 64:5 * 64]
                nbq = convqn @ prevBboardList[10 * 64:11 * 64]
                nwp = convwp @ prevBboardList[0:64]
                nbp = convbp @ prevBboardList[6 * 64:7 * 64]

                npieces = [nwkg, nbkg, nwkn, nbkn, nwbs, nbbs, nwrk, nbrk, nwq, nbq, nwp, nbp]

                res = self.directInfluence(res, listDiff, pivot, npieces, convkn, convbp, convwp, convbis, convrk,
                                           convqn, King, evanescentKing, kingColor, colorMoved)

                res = self.indirectInfluence(res, prevBboardList, prevSponge, superimposedList, pivot, k, convbis,
                                             convrk, dirk1, dirk2, dirb1, dirb2, npieces, King, evanescentKing,
                                             colorMoved)

        return res

    def indirectInfluence(self, res, prevBboardList, prevSponge, superimposedList, pivot, k, convbis, convrk, dirk1,
                          dirk2, dirb1, dirb2, npieces, King, evanescentKing, colorMoved):
        # print("indirect influence")

        nwkg = npieces[0]
        nbkg = npieces[1]
        nwbs = npieces[4]
        nbbs = npieces[5]
        nwrk = npieces[6]
        nbrk = npieces[7]
        nwq = npieces[8]
        nbq = npieces[9]

        if (nwkg == 1 and not colorMoved and superimposedList[
            k] == -1):  # white king loses one protection due to indirect influence of black attacking piece
            res -= nwkg * a9
        if (nbkg == 1 and colorMoved and superimposedList[
            k] == -1):  # black king loses one protection due to indirect influence of white attacking piece
            res += nbkg * a9
        if (nwbs >= 1 and superimposedList[k] == 0):
            maskedPrev = []
            for i in range(12):
                maskedPrev.extend(prevBboardList[i * 64:(i + 1) * 64] * convbis)
            maskedPrev.extend([0] * 5)
            maskedPrev = np.asarray(maskedPrev)
            if (superimposedList[k] == 0 and (pivot > 0 or (King and not evanescentKing))):
                res -= a2 * self.subBishopCtlBf(True, maskedPrev, self.getList()[2 * 64:3 * 64] * convbis)
            elif (pivot < 0 or (King and evanescentKing)):
                res += a2 * self.subBishopCtlBf(True, maskedPrev, self.getList()[2 * 64:3 * 64] * convbis)
            # print("res wbp influence: ", res)
        if (nbbs >= 1 and superimposedList[k] == 0):
            maskedPrev = []
            for i in range(12):
                maskedPrev.extend(prevBboardList[i * 64:(i + 1) * 64] * convbis)
            maskedPrev.extend([0] * 5)
            maskedPrev = np.asarray(maskedPrev)
            if (pivot > 0 or (King and not evanescentKing)):
                res -= -a2 * self.subBishopCtlBf(False, maskedPrev, self.getList()[8 * 64:9 * 64] * convbis)
            elif (pivot < 0 or (King and evanescentKing)):
                res += -a2 * self.subBishopCtlBf(False, maskedPrev, self.getList()[8 * 64:9 * 64] * convbis)
            # print("res bbp influence: ", res)
        if (nwrk >= 1 and superimposedList[k] == 0):
            maskedPrev = []
            for i in range(12):
                maskedPrev.extend(prevBboardList[i * 64:(i + 1) * 64] * convrk)
            maskedPrev.extend([0] * 5)
            maskedPrev = np.asarray(maskedPrev)
            if (pivot > 0 or (King and not evanescentKing)):
                res -= a4 * self.subRookCtlBf(True, maskedPrev, self.__list[3 * 64:4 * 64] * convrk)
            if (pivot < 0 or (King and evanescentKing)):
                res += a4 * self.subRookCtlBf(True, maskedPrev, self.__list[3 * 64:4 * 64] * convrk)
                # print("res wrk influence: ", res)
        if (nbrk >= 1 and superimposedList[k] == 0):
            maskedPrev = []
            for i in range(12):
                maskedPrev.extend(prevBboardList[i * 64:(i + 1) * 64] * convrk)
            maskedPrev.extend([0] * 5)
            maskedPrev = np.asarray(maskedPrev)
            if (pivot > 0 or (King and not evanescentKing)):
                res -= -a4 * self.subRookCtlBf(False, maskedPrev, self.__list[9 * 64:10 * 64] * convrk)
            elif (pivot < 0 or (King and evanescentKing)):
                res += -a4 * self.subRookCtlBf(False, maskedPrev, self.__list[9 * 64:10 * 64] * convrk)
                # print("res brk influence: ", res)
        if (nwq >= 1 and superimposedList[k] == 0):
            speconv = np.asarray([0] * 64)
            if np.sum(convbis @ prevBboardList[4 * 64:5 * 64]) == 1:  # pivot on queen's diagonal
                # print("# pivot on queen's diagonal")
                for d in dirb1:
                    if np.sum(d[0][k] @ prevBboardList[4 * 64:5 * 64]) == 1:  # queen on NE SW direction
                        # print("# queen on NE SW direction")
                        dir = dirb1
                        break
                    else:  # queen on NW SE direction
                        # print("# queen on NW SE direction")
                        dir = dirb2
            else:  # pivot on queen's horizontal/vertical
                # print("# pivot on queen's horizontal/vertical")
                for d in dirk1:
                    if np.sum(d[0][k] @ prevBboardList[4 * 64:5 * 64]) == 1:  # queen on S N direction
                        # print("# queen on S N direction")
                        dir = dirk1
                        break
                    else:  # queen on W E direction
                        # print("# queen on W E direction")
                        dir = dirk2
            for d in dir:
                j = 0
                while (np.sum(d[-j][k] @ prevSponge) > 1):
                    j += 1
                speconv += d[-j][k]
                # #print("speconv: ", speconv)
            maskedPrev = []
            for i in range(12):
                maskedPrev.extend(prevBboardList[i * 64:(i + 1) * 64] * speconv)
            maskedPrev.extend([0] * 5)
            maskedPrev = np.asarray(maskedPrev)
            if (pivot > 0 or (King and not evanescentKing)):
                res -= a3 * self.subQueenCtlBf(True, maskedPrev, self.__list[4 * 64:5 * 64])
            elif (pivot < 0 or (King and evanescentKing)):
                res += a3 * self.subQueenCtlBf(True, maskedPrev, self.__list[4 * 64:5 * 64])
                # print("res wq influence: ", res)
        if (nbq >= 1 and superimposedList[k] == 0):
            speconv = np.asarray([0] * 64)
            if np.sum(convbis @ prevBboardList[10 * 64:11 * 64]) == 1:  # pivot on queen's diagonal
                for d in dirb1:
                    if np.sum(d[0][k] @ prevBboardList[10 * 64:11 * 64]) == 1:  # queen on NE SW direction
                        dir = dirb1
                        break
                    else:  # queen on NW SE direction
                        dir = dirb2
            else:  # pivot on queen's horizontal/vertical
                for d in dirk1:
                    if np.sum(d[0][k] @ prevBboardList[10 * 64:11 * 64]):  # queen on S N direction
                        dir = dirk1
                        break
                    else:  # queen on W E direction
                        dir = dirk2
            for d in dir:
                j = 0
                while (np.sum(d[-j][k] @ prevSponge) > 1):
                    j += 1
                speconv += d[-j][k]
            # #print("speconv: ", speconv)
            maskedPrev = []
            for i in range(12):
                maskedPrev.extend(prevBboardList[i * 64:(i + 1) * 64] * speconv)
            maskedPrev.extend([0] * 5)
            maskedPrev = np.asarray(maskedPrev)
            if (pivot > 0 or (King and not evanescentKing)):
                res -= -a3 * self.subQueenCtlBf(False, maskedPrev, self.__list[10 * 64:11 * 64])
            elif (pivot < 0 or (King and evanescentKing)):
                res += -a3 * self.subQueenCtlBf(False, maskedPrev, self.__list[10 * 64:11 * 64])

                # print("res bq influence: ", res)
        return res

    def directInfluence(self, res, listDiff, pivot, npieces, convkn, convbp, convwp, convbis, convrk, convqn, King,
                        evanescentKing, kingColor, colorMoved):
        # print("direct influence")

        nwkg = npieces[0]
        nbkg = npieces[1]
        nwkn = npieces[2]
        nbkn = npieces[3]
        nwbs = npieces[4]
        nbbs = npieces[5]
        nwrk = npieces[6]
        nbrk = npieces[7]
        nwq = npieces[8]
        nbq = npieces[9]
        nwp = npieces[10]
        nbp = npieces[11]

        if (pivot != 0):
            t1 = np.sign(pivot) * nwkg if colorMoved else -np.sign(pivot) * nbkg
        else:  # equiXchange, pivot = 0 but king may have gained a protection
            t1 = nwkg if colorMoved else nbkg

        res += a9 * t1  # king wins/loses one protection due to direct influence of allied neighbouring piece
        # print("res king protected: ", res)

        # #print("king color: ", kingColor)

        nbpieces = a5 * nbkn + a2 * nbbs + a4 * nbrk + a3 * nbq + a6 * nbp
        nwpieces = a5 * nwkn + a2 * nwbs + a4 * nwrk + a3 * nwq + a6 * nwp
        # print("nbpieces: ", nbpieces, 'nwpieces: ', nwpieces)
        # print("pivot: ", pivot)

        if (King == True):
            if (evanescentKing == False):
                res -= -9 * nbpieces if kingColor == True else 9 * nwpieces
            elif (evanescentKing == True):
                corr = 0
                if (kingColor == True):
                    wkgList = (listDiff[5 * 64:6 * 64] + abs(listDiff[5 * 64:6 * 64])) / 2
                    if nbbs >= 1:
                        corr += a2 * (wkgList * listDiff[7 * 64:8 * 64]) @ convbis
                    if nbq >= 1:
                        corr += a3 * (wkgList * listDiff[10 * 64:11 * 64]) @ convqn
                    if nbrk >= 1:
                        corr += a4 * (wkgList * listDiff[9 * 64:10 * 64]) @ convrk
                    if nbp >= 1:
                        corr += a6 * (wkgList * listDiff[6 * 64:7 * 64]) @ convbp
                    nbpieces += corr
                else:
                    bkgList = (listDiff[11 * 64:12 * 64] + abs(listDiff[11 * 64:12 * 64])) / 2
                    if nwbs >= 1:
                        corr += a2 * (bkgList * listDiff[1 * 64:2 * 64]) @ convbis
                    if nwq >= 1:
                        corr += a3 * (bkgList * listDiff[4 * 64:5 * 64]) @ convqn
                    if nwrk >= 1:
                        corr += a4 * (bkgList * listDiff[3 * 64:4 * 64]) @ convrk
                    if nwp >= 1:
                        corr += a6 * (bkgList * listDiff[0 * 64:1 * 64]) @ convwp
                    nwpieces += corr
                res -= -9 * nbpieces if kingColor == True else 9 * nwpieces
                # print("res king leaves exposure: ", res)

        if (pivot > 0):
            res += pivot * a5 * (nwkn - nbkn)
            res += pivot * a2 * (nwbs - nbbs)
            res += pivot * a4 * (nwrk - nbrk)
            res += pivot * a3 * (nwq - nbq)
            res += pivot * a6 * (nwp - nbp)
        if (pivot < 0):
            listPos = (listDiff + abs(listDiff)) / 2
            for i in range(len(listPos) - 5):
                if listPos[i] != 0:
                    j = floor(i / 64)
                    List = listPos[j * 64:(j + 1) * 64]
                    colorPivot = True if i < 6 * 64 else False
            corr = 0
            if (colorPivot == True):
                if nbbs >= 1:
                    corr += a2 * (List * listDiff[8 * 64:9 * 64]) @ convbis
                if nbq >= 1:
                    corr += a3 * (List * listDiff[10 * 64:11 * 64]) @ convqn
                if nbrk >= 1:
                    corr += a4 * (List * listDiff[9 * 64:10 * 64]) @ convrk
                if nbp >= 1:
                    corr += a6 * (List * listDiff[6 * 64:7 * 64]) @ convbp
                if nbkn >= 1:
                    corr += a5 * (List * listDiff[7 * 64:8 * 64]) @ convkn
                nbpieces += corr
            else:
                if nwbs >= 1:
                    corr += a2 * (List * listDiff[2 * 64:3 * 64]) @ convbis
                if nwq >= 1:
                    corr += a3 * (List * listDiff[4 * 64:5 * 64]) @ convqn
                if nwrk >= 1:
                    corr += a4 * (List * listDiff[3 * 64:4 * 64]) @ convrk
                if nwp >= 1:
                    corr += a6 * (List * listDiff[0 * 64:1 * 64]) @ convwp
                if nwkn >= 1:
                    corr += a5 * (List * listDiff[1 * 64:2 * 64]) @ convkn
                nwpieces += corr
            # print("corr: ",corr)
            res += pivot * (nwpieces - nbpieces)
            # print('resDirectInfluence: ', res)
        return res

    def pawnPenalty(self):
        return self.subPawnPenalty(True, self.__list.copy()) - self.subPawnPenalty(False, self.__list.copy())

    def kingProtected(self):
        return self.subKingProtected(True, self.__list.copy()) - self.subKingProtected(False, self.__list.copy())

    def pawnCtl(self):
        return self.subPawnCtl(True, self.__list.copy()) - self.subPawnCtl(False, self.__list.copy())

    def knightCtl(self):
        return self.subKnightCtl(True, self.__list.copy()) - self.subKnightCtl(False, self.__list.copy())

    def rookCtl(self):
        return self.subRookCtl(True, self.__list.copy()) - self.subRookCtl(False, self.__list.copy())

    def queenCtl(self):
        return self.subQueenCtl(True, self.__list.copy()) - self.subQueenCtl(False, self.__list.copy())

    def bishopCtl(self):
        return self.subBishopCtl(True, self.__list.copy()) - self.subBishopCtl(False, self.__list.copy())

    def rookCtlBf(self):
        return self.subRookCtlBf(True, self.__list.copy()) - self.subRookCtlBf(False, self.__list.copy())

    def queenCtlBf(self):
        return self.subQueenCtlBf(True, self.__list.copy()) - self.subQueenCtlBf(False, self.__list.copy())

    def bishopCtlBf(self):
        return self.subBishopCtlBf(True, self.__list.copy()) - self.subBishopCtlBf(False, self.__list.copy())

    def subKingProtected(self, color, list, King=None):
        lst = list.copy()
        if (color == True):
            lst[6 * 64:12 * 64] = np.zeros(6 * 64)
            if (King is None):
                King = lst[5 * 64:6 * 64]
        else:
            lst[0:6 * 64] = np.zeros(6 * 64)
            if (King is None):
                King = lst[11 * 64:12 * 64]
        sponge = self.sponge(lst)
        conv = np.zeros(64)
        for i in range(64):
            if (King[i] == 1):
                conv += self.strEl.Kg[i]
            elif (King[i] == -1):
                conv -= self.strEl.Kg[i]
        return sponge @ conv

    def subPawnCtl(self, color, lst, pawns=None):
        weighted = self.getWeighted(lst)
        if (color == True):
            advKing = lst[11 * 64:12 * 64]
            if (pawns is None):
                pawns = lst[0:64]
        else:
            advKing = lst[5 * 64:6 * 64]
            if (pawns is None):
                pawns = lst[6 * 64:7 * 64]

        sponge = abs(self.sponge(weighted)) + 9 * advKing
        conv = np.asarray(64 * [0])

        for i in range(64):
            if (pawns[i] == 1 and color == True):
                conv += self.strEl.wP[i]
            elif (pawns[i] == 1 and color == False):
                conv += self.strEl.bP[i]
            elif (pawns[i] == -1 and color == True):
                conv -= self.strEl.wP[i]
            elif (pawns[i] == -1 and color == False):
                conv -= self.strEl.bP[i]

        return sponge @ conv

    def subPawnPenaltyDel(self, color, lst, pawns=None):
        if (color == True):
            sponge = lst[0:64]
            if (pawns is None):
                pawns = sponge
        else:
            sponge = lst[6 * 64:7 * 64]
            if (pawns is None):
                pawns = sponge

        conv = np.asarray(64 * [0])

        for i in range(64):
            if (pawns[i] == 1):
                conv += self.strEl.Pp[i]
            elif (pawns[i] == -1):
                conv -= self.strEl.Pp[i]

        return 2 * sponge @ conv

    def subPawnPenalty(self, color, lst):
        if (color == True):
            sponge = lst[0:64]
            pawns = sponge
        else:
            sponge = lst[6 * 64:7 * 64]
            pawns = sponge

        conv = np.asarray(64 * [0])

        for i in range(64):
            if (pawns[i] == 1):
                conv += self.strEl.Pp[i]

        return sponge @ conv

    def subBishopCtlBf(self, color, lst, bishops=None):
        weighted = self.getWeighted(lst)
        if (color == True):
            advKing = lst[11 * 64:12 * 64]
            if (bishops is None):
                bishops = lst[2 * 64:3 * 64]
        else:
            advKing = lst[5 * 64:6 * 64]
            if (bishops is None):
                bishops = lst[8 * 64:9 * 64]

        sponge = abs(self.sponge(weighted)) + 9 * advKing
        conv = np.asarray(64 * [0])

        for i in range(64):
            if (bishops[i] == 1):
                conv += self.strEl.Bp[i]
            elif (bishops[i] == -1):
                conv -= self.strEl.Bp[i]

        return sponge @ conv

    def subQueenCtlBf(self, color, lst, queen=None):
        weighted = self.getWeighted(lst)
        if (color == True):
            advKing = lst[11 * 64:12 * 64]
            if (queen is None):
                queen = lst[4 * 64:5 * 64]
        else:
            advKing = lst[5 * 64:6 * 64]
            if (queen is None):
                queen = lst[10 * 64:11 * 64]

        sponge = abs(self.sponge(weighted)) + 9 * advKing

        conv = np.asarray(64 * [0])

        for i in range(64):
            if (queen[i] == 1):
                conv += self.strEl.Qn[i]
            elif (queen[i] == -1):
                conv -= self.strEl.Qn[i]

        return sponge @ conv

    def subRookCtlBf(self, color, lst, rooks=None):
        weighted = self.getWeighted(lst)
        if (color == True):
            advKing = lst[11 * 64:12 * 64]
            if (rooks is None):
                rooks = lst[3 * 64:4 * 64]
        else:
            advKing = lst[5 * 64:6 * 64]
            if (rooks is None):
                rooks = lst[9 * 64:10 * 64]

        sponge = abs(self.sponge(weighted)) + 9 * advKing

        conv = np.asarray(64 * [0])

        for i in range(64):
            if (rooks[i] == 1):
                conv += self.strEl.Rk[i]
            elif (rooks[i] == -1):
                conv -= self.strEl.Rk[i]

        return sponge @ conv

    def subBishopCtl(self, color, lst, bishops=None):
        unweightspg = self.sponge(lst)
        weighted = self.getWeighted(lst)
        if (color == True):
            if (bishops is None):
                bishops = lst[2 * 64:3 * 64]
            advKing = lst[11 * 64:12 * 64]
        else:
            if (bishops is None):
                bishops = lst[8 * 64:9 * 64]
            advKing = lst[5 * 64:6 * 64]
        sponge = abs(self.sponge(weighted)) + 9 * advKing
        conv = np.asarray(64 * [0])
        dir = [self.strEl.NE, self.strEl.NW, self.strEl.SE, self.strEl.SW]
        for i in range(64):
            if (bishops[i] == 1):
                for d in dir:
                    j = 0
                    cond = np.sum(d[-j][i] @ unweightspg)
                    while (cond > 1):
                        j += 1
                        try:
                            cond = np.sum(d[-j][i] @ unweightspg)
                        except IndexError:
                            # print("j: ", j)
                            # print("i: ", i)
                            # print("d[-j][i]: ",d[-j+1][i])
                            # print("unweightspg: ", unweightspg)
                            raise IndexError
                    conv += d[-j][i]
            elif (bishops[i] == -1):
                for d in dir:
                    j = 0
                    cond = np.sum(d[-j][i] @ unweightspg)
                    while (cond > 1):
                        j += 1
                        try:
                            cond = np.sum(d[-j][i] @ unweightspg)
                        except IndexError:
                            # print("j: ", j)
                            # print("i: ", i)
                            # print("d[-j][i]: ",d[-j+1][i])
                            # print("unweightspg: ", unweightspg)
                            raise IndexError
                    conv -= d[-j][i]
        return sponge @ conv

    def subQueenCtl(self, color, lst, queen=None):
        unweightspg = self.sponge(lst)
        weighted = self.getWeighted(lst)
        if (color == True):
            if (queen is None):
                queen = lst[4 * 64:5 * 64]
            advKing = lst[11 * 64:12 * 64]
        else:
            if (queen is None):
                queen = lst[10 * 64:11 * 64]
            advKing = lst[5 * 64:6 * 64]
        sponge = abs(self.sponge(weighted)) + 9 * advKing
        conv = np.asarray(64 * [0])
        dir = [self.strEl.NE, self.strEl.NW, self.strEl.SE, self.strEl.SW, self.strEl.S, self.strEl.N, self.strEl.E,
               self.strEl.W]
        for i in range(64):
            if (queen[i] == 1):
                for d in dir:
                    j = 0
                    cond = np.sum(d[-j][i] @ unweightspg)
                    while (cond > 1):
                        j += 1
                        cond = d[-j][i] @ unweightspg
                    conv += d[-j][i]
            elif (queen[i] == -1):
                for d in dir:
                    j = 0
                    cond = d[-j][i] @ unweightspg
                    while (cond > 1):
                        j += 1
                        cond = d[-j][i] @ unweightspg
                    conv -= d[-j][i]
        return sponge @ conv

    def subRookCtl(self, color, lst, rooks=None):
        unweightspg = self.sponge(lst)
        weighted = self.getWeighted(lst)
        if (color == True):
            if (rooks is None):
                rooks = lst[3 * 64:4 * 64]
            advKing = lst[11 * 64:12 * 64]
        else:
            if (rooks is None):
                rooks = lst[9 * 64:10 * 64]
            advKing = lst[5 * 64:6 * 64]
        sponge = abs(self.sponge(weighted)) + 9 * advKing
        conv = np.asarray(64 * [0])
        dir = [self.strEl.S, self.strEl.N, self.strEl.E, self.strEl.W]
        for i in range(64):
            if (rooks[i] == 1):
                for d in dir:
                    j = 0
                    cond = np.sum(d[-j][i] @ unweightspg)
                    while (cond > 1):
                        j += 1
                        try:
                            cond = np.sum(d[-j][i] @ unweightspg)
                        except IndexError:
                            # print("j: ", j)
                            # print("i: ", i)
                            # print("d[-j][i]: ",d[-j+1][i])
                            # print("unweightspg: ", unweightspg)
                            raise IndexError
                    conv += d[-j][i]
            elif (rooks[i] == -1):
                for d in dir:
                    j = 0
                    cond = np.sum(d[-j][i] @ unweightspg)
                    while (cond > 1):
                        j += 1
                        try:
                            cond = np.sum(d[-j][i] @ unweightspg)
                        except IndexError:
                            # print("j: ", j)
                            # print("i: ", i)
                            # print("d[-j][i]: ",d[-j+1][i])
                            # print("unweightspg: ", unweightspg)
                            raise IndexError
                    conv -= d[-j][i]
        return sponge @ conv

    def subKnightCtl(self, color, lst, knights=None):
        weighted = self.getWeighted(lst)
        if (color == True):
            if (knights is None):
                knights = lst[64:2 * 64]
            advKing = lst[11 * 64:12 * 64]
        else:
            if (knights is None):
                knights = lst[7 * 64:8 * 64]
            advKing = lst[5 * 64:6 * 64]
        sponge = abs(self.sponge(weighted)) + 9 * advKing
        conv = np.asarray(64 * [0])
        for i in range(64):
            if (knights[i] == 1):
                conv += self.strEl.Kn[i]
            elif (knights[i] == -1):
                conv -= self.strEl.Kn[i]

        return sponge @ conv

    def sponge(self, array):
        res = [0] * 64
        for i in range(12):
            res = res + array[i * 64:(i + 1) * 64]
        return res

    def setList(self, board):
        # Dictionary to map piece colors to their index for bitboard representation
        self.board = board
        self.__list = np.asarray([0b0] * (768 + 5))
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if (piece != None):
                index = square + (piece.piece_type - 1) * 64 + COLOR_OFFSET[piece.color] * 64 * 6
                self.__list[index] = 0b1

        self.__list[-5] = BOOL_BIN[board.turn]
        self.__list[-4] = BOOL_BIN[board.has_kingside_castling_rights(chess.WHITE)]
        self.__list[-3] = BOOL_BIN[board.has_queenside_castling_rights(chess.WHITE)]
        self.__list[-2] = BOOL_BIN[board.has_kingside_castling_rights(chess.BLACK)]
        self.__list[-1] = BOOL_BIN[board.has_queenside_castling_rights(chess.BLACK)]

    def setWeights(self):
        self.__pieceswght[0:64] = np.asarray([WEIGHTS['PAWN_VALUE']] * 64)
        self.__pieceswght[64:128] = np.asarray([WEIGHTS['KNIGHT_VALUE']] * 64)
        self.__pieceswght[128:192] = np.asarray([WEIGHTS['BISHOP_VALUE']] * 64)
        self.__pieceswght[192:256] = np.asarray([WEIGHTS['ROOK_VALUE']] * 64)
        self.__pieceswght[256:320] = np.asarray([WEIGHTS['QUEEN_VALUE']] * 64)
        self.__pieceswght[320:384] = np.asarray([0] * 64)
        self.__pieceswght[384:448] = np.asarray([WEIGHTS['PAWN_VALUE']] * 64)
        self.__pieceswght[448:512] = np.asarray([WEIGHTS['KNIGHT_VALUE']] * 64)
        self.__pieceswght[512:576] = np.asarray([WEIGHTS['BISHOP_VALUE']] * 64)
        self.__pieceswght[576:640] = np.asarray([WEIGHTS['ROOK_VALUE']] * 64)
        self.__pieceswght[640:704] = np.asarray([WEIGHTS['QUEEN_VALUE']] * 64)
        self.__pieceswght[704:773] = np.asarray([0] * 69)
