import chess
import chess.svg
from random import shuffle
from lexwolf.core import DummyLexWolf, LexWolfCore
from IPython.display import display, SVG


class Game():
    """
    Handles the chess games
    """
    def __init__(self, p1_is_human=True, p2_is_human=False, AIwhite=DummyLexWolf(), AIblack=DummyLexWolf(),
                 max_moves=None, verbose=1, silence=False):
        self.p1_is_human = p1_is_human
        self.p2_is_human = p2_is_human
        self.AIwhite = AIwhite  # type LexWolfCore
        self.AIblack = AIblack
        self.board = chess.Board()
        self.move_memory = []
        self.max_moves = max_moves  # stops the game if > max_moves, inactive if set on "None"
        self.move_count = 0
        self.result = 0  # -1 --> black, 0 --> draw, 1 --> white
        self.result_cause = 'Checkmate'
        self.verbose = verbose
        self.silence = silence

        self.start()  # starts the game

    def AImove(self, AI):
        next_move = AI.find_optimal_move(self.board)
        if next_move in self.board.legal_moves:
            self.load_move_in_memory(next_move)
            self.play_move(next_move)
            self.verbose_message(self.generate_AI_reaction())
        else:
            print(self.board)
            print(next_move)
            raise ValueError("The AI just played an illegal move.")

    def check_endgame(self):
        if self.board.is_checkmate():
            return 1
        elif self.board.is_stalemate() or self.board.is_fivefold_repetition() or self.board.is_seventyfive_moves()\
                or self.board.is_insufficient_material() or self.board.can_claim_draw():
            return -1
        else:
            return 0

    def human_move(self):
        self.show_board(size=500)
        valid = False
        while not valid:
            uci = input("\nYour turn: ")
            try:
                move = chess.Move.from_uci(uci)
                assert(move in self.board.legal_moves)
                self.load_move_in_memory(move)
                self.play_move(move)
                valid = True
                self.verbose_message("\nYour move is valid. Waiting for opponent...")
            except:
                self.verbose_message(f"This move is not legal. Legal moves are: {list(self.board.legal_moves)}")

    def generate_AI_reaction(self):
        reactions = ["The AI played its move.",
                     "The AI played its move. Fear, puny human!",
                     "The AI has established a strategy to crush you.",
                     "AI: hmmm... let's see how you will handle that.",
                     "AI: I played my move. And I didn't ask to GPT."]
        reactions += ["The AI played its move."] * 10
        shuffle(reactions)
        return reactions[0]

    def load_move_in_memory(self, move):
        self.move_memory.append(move)

    def message(self, mes):
        if not self.silence:
            print(mes)

    def play_move(self, move):
        if move in self.board.legal_moves:
            self.board.push(move)
        else:
            raise ValueError("Illegal move entered in 'Game.play_move'.")

    def start(self):
        self.message("\n\nTHE GAME HAS STARTED. GOOD LUCK!\n")
        while True:
            self.move_count += 1
            # White move
            if self.p1_is_human:
                self.human_move()
            else:
                self.AImove(self.AIwhite)
            if self.check_endgame() == 1:
                # White win
                self.message(f"Checkmate at move {self.move_count}")
                self.result = 1
                break
            elif self.check_endgame() == -1:
                # Draw
                self.result = 0
                break

            # Black move
            if self.p2_is_human:
                self.human_move()
            else:
                self.AImove(self.AIblack)
            if self.check_endgame() == 1:
                # Black win
                self.message(f"Checkmate at move {self.move_count}")
                self.result = -1
                break
            elif self.check_endgame() == -1:
                # Stalemate --> draw
                self.result = 0
                break

            if self.max_moves is not None and self.move_count > self.max_moves:
                self.verbose_message("EXCEEDED MAX ALLOWED MOVES FOR THIS GAME.")
                break

        self.show_board()

        if self.result == 0:
            if self.board.is_insufficient_material():
                self.message(f"Draw for insufficient material at move {self.move_count}")
                self.result_cause = "Insufficient material"
            elif self.board.is_stalemate():
                self.message(f"Draw for stalemate at move {self.move_count}")
                self.result_cause = "Stalemate"
            elif self.board.is_fivefold_repetition():
                self.message(f"Draw for fivefold repetition at move {self.move_count}")
                self.result_cause = "Fivefold repetition"
            elif self.board.can_claim_draw():
                self.message(f"Draw claimed or threefold repetition at move {self.move_count}")
                self.result_cause = "Threefold repetition"
            elif self.board.is_seventyfive_moves():
                self.message(f"Draw according to the 75-moves law at move {self.move_count}")
                self.result_cause = "75 moves"


        self.verbose_message("\nYou can visualize the full gameplay by calling 'Game.show_game()'.")
        self.message(f"Result: {self.result}")
        self.message("\nGAME OVER.")

    def show_board(self, svg_board=True, size=200):
        if svg_board:
            svg = chess.svg.board(self.board, size=size)  # Generate SVG for the current board
            display(SVG(svg))  # Display the SVG in Jupyter Notebook
        else:
            self.verbose_message(self.board)  # Fallback to verbose message if SVG not desired

    def show_game(self):
        self.board = chess.Board()
        for move in self.move_memory:
            self.play_move(move)
            self.message("")
            self.show_board()

    def verbose_message(self, message):
        if self.verbose:
            self.message(message)