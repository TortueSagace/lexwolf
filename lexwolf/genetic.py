from time import time

import chess
from random import shuffle, seed, randrange, choices, random
import numpy as np

from lexwolf.core import LexWolfCore


class GeneticLexWolf(LexWolfCore):
    def __init__(self, population_size=10, mutation_rate=0.01, max_generations=100, **kwargs):
        super().__init__(**kwargs)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.population = self.initialize_population()

    def initialize_population(self):
        """
        Initialize a population of random parameter sets for the evaluation function.
        Each parameter set is a dictionary representing weights for various aspects of
        a chess position, which are essential for evaluating the position's strength.
        """
        return [self.random_parameters() for _ in range(self.population_size)]

    def random_parameters(self):
        """
        Generate a comprehensive and random set of parameters for the evaluation function.
        These parameters include weights for various strategic elements of chess, ensuring
        a wide exploration space for the genetic algorithm.
        """
        return {
            'pawn_value': np.random.uniform(1, 1.2),  # Value of a pawn
            'knight_value': np.random.uniform(3, 3.2),  # Value of a knight
            'bishop_value': np.random.uniform(3, 3.2),  # Value of a bishop
            'rook_value': np.random.uniform(5, 5.2),  # Value of a rook
            'queen_value': np.random.uniform(9, 9.2),  # Value of a queen
            'material_weight': np.random.uniform(0.8, 1.2),  # Overall material balance
            'mobility_weight': np.random.uniform(0.1, 0.3),  # Mobility of pieces
            'center_control_weight': np.random.uniform(0.1, 0.3),  # Control of the center
            'pawn_structure_weight': np.random.uniform(0.1, 0.3),  # Pawn structure health
            'king_safety_weight': np.random.uniform(0.1, 0.3),  # Safety of the king
            'bishop_pair_weight': np.random.uniform(0.1, 0.3),  # Bonus for having both bishops
            'rook_on_open_file_weight': np.random.uniform(0.1, 0.3),  # Rook(s) on open files
            'passed_pawn_weight': np.random.uniform(0.1, 0.3),  # Advantages of passed pawns
            'isolated_pawn_penalty': np.random.uniform(-0.3, -0.1),  # Penalty for isolated pawns
            'doubled_pawn_penalty': np.random.uniform(-0.3, -0.1),  # Penalty for doubled pawns
            'knight_outpost_weight': np.random.uniform(0.1, 0.3),  # Knights on strong outposts
            'queen_activity_weight': np.random.uniform(0.1, 0.3),  # Activity level of the queen
        }

    def evaluate_fitness(self, parameters, board_states):
        """
        Evaluate the fitness of a set of parameters by analyzing multiple board states.
        The fitness score is calculated based on the cumulative evaluation scores from
        a predefined set of board states, representing different stages or challenges
        in a chess game. This method simulates how well the parameters perform across
        various scenarios, rather than relying on a single board state.

        :param parameters: The set of parameters to evaluate.
        :param board_states: A list of chess.Board() instances representing different game states.
        :return: The average score of the parameters across all provided board states, representing the fitness.
        """
        board_states = self.generate_board_states()  # Generate or load diverse game scenarios
        total_score = 0
        for board in board_states:
            score = self.evaluate_board(board, parameters)
            total_score += score

        average_score = total_score / len(board_states)
        return average_score

    def evaluate_board(self, board, parameters):
        """
        Evaluates the chess board using evolved parameters to score based on material,
        mobility, board control, pawn structure, king safety, and other strategic elements.
        """
        score = 0

        # Material score
        for piece_type, value_key in zip([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN],
                                         ['pawn_value', 'knight_value', 'bishop_value', 'rook_value', 'queen_value']):
            score += (len(board.pieces(piece_type, chess.WHITE)) - len(board.pieces(piece_type, chess.BLACK))) * \
                     parameters[value_key]

        # Mobility score
        mobility_score = (len(list(board.legal_moves))) * parameters['mobility_weight']
        score += mobility_score

        # Center control
        center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
        center_control_score = sum(1 for move in board.legal_moves if move.to_square in center_squares) * parameters[
            'center_control_weight']
        score += center_control_score

        # Pawn structure (penalties for isolated and doubled pawns)
        isolated_pawns = self.calculate_isolated_pawns(board)
        doubled_pawns = self.calculate_doubled_pawns(board)
        score += isolated_pawns * parameters['isolated_pawn_penalty']
        score += doubled_pawns * parameters['doubled_pawn_penalty']

        # King safety (a simplistic approach could be the distance of the king from the edge of the board)
        king_safety_score = self.evaluate_king_safety(board, parameters)
        score += king_safety_score * parameters['king_safety_weight']

        return score

    def calculate_isolated_pawns(self, board):
        """
        Calculate the number of isolated pawns for both sides and return the difference.
        Isolated pawns are those with no friendly pawns on their adjacent files.
        """
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)
        white_isolated = 0
        black_isolated = 0

        for pawn in white_pawns:
            if not self.has_adjacent_pawn(pawn, white_pawns, board):
                white_isolated += 1

        for pawn in black_pawns:
            if not self.has_adjacent_pawn(pawn, black_pawns, board):
                black_isolated += 1

        # Return the difference in the number of isolated pawns (white - black)
        return white_isolated - black_isolated

    def has_adjacent_pawn(self, square, pawns, board):
        """
        Check if a given pawn has friendly pawns on its adjacent files.
        """
        file = chess.square_file(square)
        rank = chess.square_rank(square)

        adjacent_files = [f for f in [file - 1, file + 1] if f >= 0 and f <= 7]

        for adj_file in adjacent_files:
            for adj_rank in range(8):
                if chess.square(adj_file, adj_rank) in pawns:
                    return True
        return False

    def calculate_doubled_pawns(self, board):
        """
        Calculate the number of doubled pawns for both sides and return the difference.
        Doubled pawns are those that share the same file with another pawn of the same color.
        """
        white_doubled = self.count_doubled_pawns(board, chess.WHITE)
        black_doubled = self.count_doubled_pawns(board, chess.BLACK)

        # Return the difference in the number of doubled pawns (white - black)
        return white_doubled - black_doubled

    def count_doubled_pawns(self, board, color):
        """
        Count doubled pawns for a given color.
        """
        pawns = board.pieces(chess.PAWN, color)
        file_count = [0] * 8  # There are 8 files on a chessboard, indexed 0-7

        for pawn in pawns:
            file = chess.square_file(pawn)
            file_count[file] += 1

        # Doubled pawns are those with file counts greater than 1
        doubled = sum(count - 1 for count in file_count if count > 1)
        return doubled

    def evaluate_king_safety(self, board, parameters):
        """
        Evaluate the king's safety based on its position, surrounding pawns, and potential threats.
        Higher scores indicate better safety.
        """
        safety_score = 0

        # Get king positions
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)

        # Evaluate pawn shield for both kings
        safety_score += self.evaluate_pawn_shield(board, white_king_square, chess.WHITE, parameters)
        safety_score -= self.evaluate_pawn_shield(board, black_king_square, chess.BLACK, parameters)

        # Potential additional evaluations can include checking open files, diagonals, etc.

        return safety_score * parameters['king_safety_weight']

    def evaluate_pawn_shield(self, board, king_square, color, parameters):
        """
        Evaluate the pawn shield in front of the king, considering pawns directly in front of
        and adjacent to the king. The evaluation gives higher importance to pawns that are
        directly in front of the king for their protective value.
        """
        pawn_shield_score = 0
        pawn_positions = board.pieces(chess.PAWN, color)

        # Adjustments to locate pawns around the king based on color
        rank_adjustments = [1, 1, 1] if color == chess.WHITE else [-1, -1, -1]
        file_adjustments = [-1, 0, 1]

        for rank_adj in rank_adjustments:
            for file_adj in file_adjustments:
                if rank_adj == 0 and file_adj == 0:
                    continue  # Skip the square where the king is located
                adjacent_square = chess.square(chess.square_file(king_square) + file_adj,
                                               chess.square_rank(king_square) + rank_adj)

                # Score adjustment for pawns directly in front of the king
                if file_adj == 0 and adjacent_square in pawn_positions:
                    pawn_shield_score += parameters.get('direct_pawn_shield_value', 2)  # Default value if not specified
                elif adjacent_square in pawn_positions:
                    pawn_shield_score += 1  # Adjacent pawns still add value but less than direct front

        return pawn_shield_score

    def evaluate_population_fitness(self, population):
        """
        Evaluate the fitness of the entire population and return a list of fitness scores.
        """
        # Placeholder for actual implementation
        return [self.evaluate_fitness(individual['parameters'], self.board_states) for individual in population]

    def select_parents(self):
        """
        Selects parents for the next generation using roulette wheel selection.
        """
        total_fitness = sum(self.fitness_scores)
        selection_probs = [fitness / total_fitness for fitness in self.fitness_scores]

        # Select two parents based on their selection probabilities
        parents = choices(self.population, weights=selection_probs, k=2)
        return parents

    # The implementation of evaluate_fitness and other existing methods...

    def find_optimal_move(self, board=chess.Board()):
        """
        Override the method to use the evolved parameters to find the optimal move.
        """
        # Example: Select the move that leads to the best score using the current best parameters
        best_move = None
        best_score = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            score = self.evaluate_board(board, self.best_parameters)
            if score > best_score:
                best_move = move
                best_score = score
            board.pop()
        return best_move

    def crossover(self, parent1, parent2):
        """
        Perform uniform crossover between two parents to produce offspring.
        Each parameter of the offspring has an equal chance of being selected
        from either parent.

        :param parent1: The first parent's parameter set.
        :param parent2: The second parent's parameter set.
        :return: Offspring's parameter set resulting from the crossover.
        """
        offspring = {}
        for key in parent1:
            # For each parameter, randomly choose the value from one of the parents
            offspring[key] = parent1[key] if random() < 0.5 else parent2[key]
        return offspring

    def mutate(self, individual):
        """
        Introduce random variations to an individual's parameters.
        The mutation rate determines the probability of each parameter being mutated.

        :param individual: The individual's parameter set to be mutated.
        """
        for key in individual.keys():
            if np.random.rand() < self.mutation_rate:
                # Apply mutation based on the type of parameter
                if 'weight' in key or 'value' in key:
                    # Assume numerical parameters are mutated by a small random factor
                    mutation_factor = np.random.uniform(0.95, 1.05)
                    individual[key] *= mutation_factor
                elif 'penalty' in key:
                    # Penalties might be mutated in a different way if needed
                    mutation_factor = np.random.uniform(0.95, 1.05)
                    individual[key] *= mutation_factor
                # Add other mutation cases as needed based on parameter names and types

        return individual

    def create_next_generation(self):
        """
        Creates the next generation by applying selection, crossover, and mutation.
        This function embodies the core evolutionary process, enabling the population
        to evolve over generations.
        """
        new_population = []
        while len(new_population) < self.population_size:
            # Select two parents from the current population
            parent1, parent2 = self.select_parents()

            # Perform crossover to produce offspring
            offspring = self.crossover(parent1, parent2)

            # Mutate the offspring
            mutated_offspring = self.mutate(offspring)

            # Add the mutated offspring to the new population
            new_population.append(mutated_offspring)

        # Update the current population with the new generation
        self.population = new_population

        # Optionally, re-evaluate the fitness of the new population here
        self.fitness_scores = self.evaluate_population_fitness(self.population)

    def generate_board_states(self):
        """
        Generate or load a diverse set of board states for fitness evaluation.
        Ideally, this method would draw from a database of game positions or generate positions
        programmatically to cover a wide range of scenarios.
        """
        board_states = []
        # Example approach: Load positions from a database or file
        # For demonstration, let's assume we manually define a few positions
        # In practice, this could involve parsing PGN files, FEN strings, or programmatically generating positions

        # Opening position
        board_states.append(chess.Board())

        # A midgame position
        midgame_fen = "r2q1rk1/ppp2ppp/2npbn2/3Np1B1/2PP4/2N1P3/PP3PPP/R2QKB1R w KQ - 0 1"
        board_states.append(chess.Board(midgame_fen))

        # An endgame position
        endgame_fen = "8/2p5/1p1p4/p2Pp3/P3Pp2/1P3Pp1/2P3P1/6K1 w - - 0 1"
        board_states.append(chess.Board(endgame_fen))

        # Additional positions could be added to cover a wide range of scenarios
        return board_states