import random

def generate_maze(n):
    """Generate a maze using Wilson's algorithm."""
    maze = [[{'N': True, 'S': True, 'E': True, 'W': True} for _ in range(n)] for _ in range(n)]
    maze_tree = set()

    # Initialize maze with a random starting cell
    start_x = random.randint(0, n - 1)
    start_y = random.randint(0, n - 1)
    maze_tree.add((start_x, start_y))

    while len(maze_tree) < n * n:
        # Select a random cell not in the maze tree
        cells_not_in_maze = [(x, y) for x in range(n) for y in range(n) if (x, y) not in maze_tree]
        current_cell = random.choice(cells_not_in_maze)

        path = []
        path_cells = set()

        # Perform random walk until reaching the maze tree
        while True:
            path.append(current_cell)
            path_cells.add(current_cell)

            if current_cell in maze_tree:
                break

            x, y = current_cell
            directions = []
            if y > 0:
                directions.append(('N', (x, y - 1)))
            if y < n - 1:
                directions.append(('S', (x, y + 1)))
            if x > 0:
                directions.append(('W', (x - 1, y)))
            if x < n - 1:
                directions.append(('E', (x + 1, y)))

            direction, next_cell = random.choice(directions)

            if next_cell in path_cells:
                # Remove loop in path
                idx = path.index(next_cell)
                path = path[:idx + 1]
                path_cells = set(path)
                current_cell = next_cell
            else:
                current_cell = next_cell

        maze_tree.update(path)

        # Remove walls along the path
        for i in range(len(path) - 1):
            cell1 = path[i]
            cell2 = path[i + 1]
            x1, y1 = cell1
            x2, y2 = cell2

            if x2 == x1 and y2 == y1 - 1:
                maze[y1][x1]['N'] = False
                maze[y2][x2]['S'] = False
            elif x2 == x1 and y2 == y1 + 1:
                maze[y1][x1]['S'] = False
                maze[y2][x2]['N'] = False
            elif x2 == x1 - 1 and y2 == y1:
                maze[y1][x1]['W'] = False
                maze[y2][x2]['E'] = False
            elif x2 == x1 + 1 and y2 == y1:
                maze[y1][x1]['E'] = False
                maze[y2][x2]['W'] = False

    # Add entrance on a random border
    entrance_side = random.choice(['N', 'S', 'E', 'W'])
    if entrance_side == 'N':
        x = random.randint(0, n - 1)
        maze[0][x]['N'] = False
        entrance = ((x, 0), 'N')
    elif entrance_side == 'S':
        x = random.randint(0, n - 1)
        maze[n - 1][x]['S'] = False
        entrance = ((x, n - 1), 'S')
    elif entrance_side == 'W':
        y = random.randint(0, n - 1)
        maze[y][0]['W'] = False
        entrance = ((0, y), 'W')
    elif entrance_side == 'E':
        y = random.randint(0, n - 1)
        maze[y][n - 1]['E'] = False
        entrance = ((n - 1, y), 'E')

    # Add exit on a different border
    exit_sides = ['N', 'S', 'E', 'W']
    exit_sides.remove(entrance_side)
    exit_side = random.choice(exit_sides)
    if exit_side == 'N':
        x = random.randint(0, n - 1)
        maze[0][x]['N'] = False
        exit = ((x, 0), 'N')
    elif exit_side == 'S':
        x = random.randint(0, n - 1)
        maze[n - 1][x]['S'] = False
        exit = ((x, n - 1), 'S')
    elif exit_side == 'W':
        y = random.randint(0, n - 1)
        maze[y][0]['W'] = False
        exit = ((0, y), 'W')
    elif exit_side == 'E':
        y = random.randint(0, n - 1)
        maze[y][n - 1]['E'] = False
        exit = ((n - 1, y), 'E')

    return maze, entrance, exit