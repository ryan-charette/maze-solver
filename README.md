# Maze Solver

This project is a Python-based implementation of maze generation, solving, and visualization using various algorithms, including **Dijkstra's algorithm** and **Deep Q-Learning (DQN)**. The project supports real-time visualization of pathfinding and provides a reinforcement learning agent to solve the maze optimally.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Maze Generation](#1-maze-generation)
  - [2. Dijkstra's Algorithm](#2-dijkstras-algorithm)
  - [3. DQN-based Maze Solving](#3-dqn-based-maze-solving)
- [Visualization](#visualization)
- [References](#references)

---

## Features

1. **Maze Generation**:
   - Uses **Wilson's algorithm** to generate perfect mazes with guaranteed path connectivity.
   - Randomized entrances and exits for increased variability.

2. **Pathfinding Algorithms**:
   - **Dijkstra's Algorithm** for shortest path finding.
   - Real-time visualization of Dijkstra's algorithm.

3. **Reinforcement Learning**:
   - Implements a **Deep Q-Network (DQN)** to learn and solve mazes.
   - Experience replay and epsilon-greedy policy for efficient training.

---

## Project Structure

```
├── dijkstra.py # Dijkstra's algorithm implementation
├── display_maze.py # Maze rendering and visualization
├── dqn_maze_solver.py # DQN-based maze solving
├── maze_generator.py # Maze generation using Wilson's algorithm
```

---

## Requirements

- Python 3.9+
- Libraries:
  - `numpy`
  - `matplotlib`
  - `torch` (PyTorch)
  - `OpenGL`
  - `glfw`
  - `gym`

You can install the necessary libraries using:

```bash
pip install numpy matplotlib torch PyOpenGL glfw gym
```

## Installation
Clone this repository and navigate to the project directory:
```bash
git clone https://github.com/your-username/maze-solver-project.git
cd maze-solver-project
```

---

## Usage

### 1. Maze Generation

To generate a maze:

```python
from maze_generator import generate_maze

n = 8  # Maze size
maze, entrance, exit = generate_maze(n)
print("Maze generated with entrance at", entrance, "and exit at", exit)
```

### 2. Dijkstra's Algorithm

To find the shortest path using Dijkstra's algorithm:

```python
from dijkstra import maze_to_graph, dijkstra
from maze_generator import generate_maze

n = 8
maze, entrance, exit = generate_maze(n)
graph = maze_to_graph(maze)
path = dijkstra(graph, entrance[0], exit[0])

print("Shortest path:", path)
```

To visualize the maze and pathfinding process:

```bash
python display_maze.py
```
This will open a window displaying the maze and the progress of Dijkstra's algorithm.

### 3. DQN-based Maze Solving

To train a DQN agent to solve the maze:

```bash
python dqn_maze_solver.py
```

This script will:

- Train a DQN agent using the generated maze.
- Display the number of steps taken in each episode.
- Save the trained model as `dqn_maze_solver.pth`.
- Plot a graph showing the learning progress.

---

## Visualization
- OpenGL-based rendering for maze and pathfinding visualization.
- Matplotlib plots for training performance of the DQN agent.

---

References
PyTorch Documentation: https://pytorch.org/docs/
OpenGL Documentation: https://www.opengl.org/documentation/
Gym Documentation: https://www.gymlibrary.dev/
Wilson's Algorithm: https://en.wikipedia.org/wiki/Maze_generation_algorithm#Wilson's_algorithm
Dijkstra's Algorithm: https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm

