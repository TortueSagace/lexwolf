import unittest
from lexwolf import Game
from lexwolf.models import DummyLexWolf, MinmaxLexWolf
from tqdm import tqdm

class TestLexWolf(unittest.TestCase):

    def test_stalemate(self):
        SEED = 0
        game = Game(False, False,
                    DummyLexWolf(random_seed=SEED), DummyLexWolf(random_seed=SEED), verbose=0)

    def test_insufficient_material(self):
        SEED = 1
        game = Game(False, False,
                    DummyLexWolf(random_seed=SEED), DummyLexWolf(random_seed=SEED), verbose=0)

    def test_75_moves(self):
        SEED = 5
        game = Game(False, False,
                    DummyLexWolf(random_seed=SEED), DummyLexWolf(random_seed=SEED), verbose=0)

    def test_checkmate(self):
        SEED = 6
        game = Game(False, False,
                    DummyLexWolf(random_seed=SEED), DummyLexWolf(random_seed=SEED), verbose=0)

    def test_Dummy_vs_Minmax(self):
        win_ratio = 0
        n_games = 100
        for SEED in tqdm(range(n_games)):
            game = Game(False, False,
                            DummyLexWolf(random_seed=SEED),
                            MinmaxLexWolf(random_seed=SEED, max_depth=2, center_bonus=0), silence=True)
            win_ratio += game.result
            print("Win ratio so far is", round(win_ratio / (SEED + 1), 2), "%")
        print(
            f"Win ratio between DummyLexWolf (white) and MinmaxLexWolf with depth 1 (black) over {n_games} games: {win_ratio}")

        # Result: -94 with center and king bonuses (0.1, 0.2)
        #         -94 with center, control and king bonuses (0.1, 0.1, 0.2)
        #         -98 with control and king bonuses (0.1, 0.2)
        #         -99 with control, check and king bonuses (0.1, 0.2, 0.2)
        #         ... with control, check and king bonuses (0.1, 0.2, 0.2) and max_depth=2

    def test_minmax_depth_efficiency(self):
        win_ratio = 0
        n_games = 100
        for i, SEED in enumerate(tqdm(range(n_games))):
            if i % 2:
                game = Game(False, False,
                            MinmaxLexWolf(random_seed=SEED, max_depth=1),
                            MinmaxLexWolf(random_seed=SEED, max_depth=2), silence=True)
            else:
                game = Game(False, False,
                            MinmaxLexWolf(random_seed=SEED, max_depth=2),
                            MinmaxLexWolf(random_seed=SEED, max_depth=1), silence=True)
            win_ratio += game.result
        print(f"Win ratio between MinmaxLexWolf d1 (white) and MinmaxLexWolf d2 (black) over {n_games} games: {win_ratio}")

"""    def test_minmax_depth_efficiency_2(self):
        win_ratio = 0
        n_games = 100
        for i, SEED in enumerate(tqdm(range(n_games))):
            if i % 2:
                game = Game(False, False,
                            MinmaxLexWolf(random_seed=SEED, max_depth=1),
                            MinmaxLexWolf(random_seed=SEED, max_depth=3), silence=True)
            else:
                game = Game(False, False,
                            MinmaxLexWolf(random_seed=SEED, max_depth=3),
                            MinmaxLexWolf(random_seed=SEED, max_depth=1), silence=True)
            win_ratio += game.result
        print(f"Win ratio between MinmaxLexWolf d1 (white) and MinmaxLexWolf d2 (black) over {n_games} games: {win_ratio}")"""


if __name__ == '__main__':
    unittest.main()
