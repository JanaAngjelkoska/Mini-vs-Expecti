# Minimax vs. Expectiminimax in Chess

![Minimax Vs  Expectiminimax](https://github.com/user-attachments/assets/2c78e2ea-7671-4456-9ae1-3aeb4d130c42)

## Project Goal
The goal of this project is to build 2 adversarial agents, heuristic functions that build an evaluation function for a small chess engine,
which is to be utilized by these algorithms and compare how they fare against each other across multiple games,  with different possible heuristics.
The end goal was to demonstrate a verification on the hypothesis that the Minimax algorithm would fare better in deterministic environments in contrast to Expectiminimax (or Expectimax)
generally, which was successfully done.
## Project Structure
Please note that the python files are well documented, and there is no documentation here.

**Directories**:
- `utils`: Utility classes and data structures that can be used for exploring how the search space of a chess environment feels like. (Note: not used in the implementation of the algorithms)
- `search`: Algorithms that play chess.
- `eval`: Definition of the evaluation function and arbitrary heuristics.
  
**Files**:
- `game.py`: Contains the code that puts both algorithms in a match, where you can choose which algorithm plays what side, and see the game progress.
