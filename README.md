# LexWolf Chess Library

## Created by Loup François and Alexandre Le Mercier

### Duration: February to May 2024

**Note: IT IS HIGHLY RECOMMENDED TO USE THIS LIBRARY IN A NOTEBOOK ENVIRONMENT!**

## Overview

The LexWolf Chess Library provides a comprehensive toolkit designed to manage and evaluate chess games through the use of advanced algorithms and robust data structures. It integrates visualization tools, evaluation mechanisms, and AI-driven decision making, leveraging the capabilities of the widely-used `python-chess` package. This library implements efficient game state representation and manipulation techniques, such as bitboards, and uses databases of opening strategies to enhance game play from the start. It is crafted for educational purposes and as a robust backend for developing more sophisticated chess-related applications.

## Table of Contents

1. **bitBoard**  
   Manages the bitboard representation of the chess board, utilizing bits to denote the presence and state of pieces. Includes methods for evaluation and move generation based on the current board state.

2. **structEl**  
   Provides structural elements for chess board representation, managing various board configurations and piece movements using matrix representations.

3. **LexWolfCore**  
   Serves as the base class for implementing various chess algorithms, with utilities for evaluating board states and generating moves. It provides a framework for extending AI capabilities.

4. **DummyLexWolf**  
   A basic AI implementation that selects moves randomly. It acts as a baseline or placeholder AI.

5. **MinmaxLexWolf**  
   Implements the Minimax algorithm specifically tailored for chess, including optimizations like alpha-beta pruning to efficiently search the game tree.

6. **AdaptativeMinmaxLexWolf**  
   An enhanced version of the Minimax AI that utilizes a database of popular and efficient opening strategies to optimize early-game moves.

7. **Game**  
   Manages a chess game between two players, which can be either human or AI-driven. This class handles game setup, move validation, game state updates, and determining game outcomes according to chess rules.

## Usage

To utilize this library, instantiate game objects, AI players, and manage game states as needed. The library supports both interactive and automated game sessions, providing flexibility for different user interactions and testing scenarios.

Example of initializing a game:
```python
from lexwolf import *
game = Game(True, False, _, AdaptativeMinmaxLexWolf(max_depth=3))
```
This line configures a game with a human player as White and an AI using the adaptative model with depth 3 as Black.

## Dependencies

- **python-chess**: For chess board management and legal move generation.
- **numpy**: For efficient numerical operations, particularly for board state evaluations.
- **pandas**: For managing opening book datasets.
- **IPython**: For displaying the boards visually in a notebook.
- **time, random, math, os**: Standard Python libraries utilized for various functional and operational purposes.

## Installation

To install the LexWolf Chess Library, ensure that you have Python installed on your system, and run the following command in your terminal:

```bash
pip install lexwolf
```

Ensure all dependencies are also installed using pip if they are not already present in your environment.

## Contributing

Contributions to the LexWolf Chess Library are welcome! Please fork the repository and submit a pull request with your proposed changes. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
