import time
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
from maze_generator import generate_maze
from dijkstra import maze_to_graph, dijkstra, dijkstra_generator

def draw_cell(position, cell_size, margin):
    """Render a single maze cell as a quadrilateral."""
    x, y = position
    x_pos = -1 + margin + x * cell_size
    y_pos = 1 - margin - (y + 1) * cell_size  # Invert y-axis
    glBegin(GL_QUADS)
    glVertex2f(x_pos, y_pos)
    glVertex2f(x_pos + cell_size, y_pos)
    glVertex2f(x_pos + cell_size, y_pos + cell_size)
    glVertex2f(x_pos, y_pos + cell_size)
    glEnd()

def get_outside_point(position, side, cell_size, margin):
    """Calculate a point outside the maze cell for entrance or exit."""
    x, y = position
    x_center = -1 + margin + (x + 0.5) * cell_size
    y_center = 1 - margin - (y + 0.5) * cell_size
    offset = cell_size / 2
    if side == 'N':
        return x_center, y_center + offset
    elif side == 'S':
        return x_center, y_center - offset
    elif side == 'E':
        return x_center + offset, y_center
    elif side == 'W':
        return x_center - offset, y_center

def draw_maze(maze, entrance, exit, algorithm_state=None):
    """Render the entire maze with optional algorithm visualization."""
    entrance_pos, entrance_side = entrance
    exit_pos, exit_side = exit
    n = len(maze)
    margin = 0.05
    cell_size = (2.0 - 2 * margin) / n
    
    inner_line_width = 1.0
    outer_line_width = inner_line_width * 3
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
        
    if algorithm_state:
        visited = algorithm_state.get('visited', set())
        heap = algorithm_state.get('heap', [])
        current_node = algorithm_state.get('current_node', None)
    
        # Draw visited cells
        glColor3f(0.8, 0.8, 0.8)
        for node in visited:
            draw_cell(node, cell_size, margin)
    
        # Draw frontier cells
        glColor3f(1.0, 1.0, 0.0)
        for _, node in heap:
            if node not in visited:
                draw_cell(node, cell_size, margin)
    
        # Highlight current node
        if current_node:
            glColor3f(0.0, 1.0, 1.0)
            draw_cell(current_node, cell_size, margin)
    
    # Draw the path if available
    if algorithm_state:
        path = algorithm_state.get('path', None)
        if path:
            entrance_outside = get_outside_point(entrance_pos, entrance_side, cell_size, margin)
            exit_outside = get_outside_point(exit_pos, exit_side, cell_size, margin)
    
            extended_path = [entrance_outside]
            for position in path:
                x, y = position
                x_center = -1 + margin + (x + 0.5) * cell_size
                y_center = 1 - margin - (y + 0.5) * cell_size
                extended_path.append((x_center, y_center))
            extended_path.append(exit_outside)
    
            glColor3f(0.0, 0.0, 1.0)
            glLineWidth(2.0)
            glBegin(GL_LINE_STRIP)
            for x_pos, y_pos in extended_path:
                glVertex2f(x_pos, y_pos)
            glEnd()
            glLineWidth(1.0)
    
    glColor3f(0.0, 0.0, 0.0)
    
    # Separate walls into inner and outer for rendering
    inner_walls = []
    outer_walls = []
    
    for y in range(n):
        for x in range(n):
            cell = maze[y][x]
    
            x_pos = -1 + margin + x * cell_size
            y_pos = 1 - margin - (y + 1) * cell_size  # Invert y-axis
    
            if cell['N']:
                start = (x_pos, y_pos + cell_size)
                end = (x_pos + cell_size, y_pos + cell_size)
                if y == 0:
                    outer_walls.append((start, end))
                else:
                    inner_walls.append((start, end))
    
            if cell['S']:
                start = (x_pos, y_pos)
                end = (x_pos + cell_size, y_pos)
                if y == n - 1:
                    outer_walls.append((start, end))
                else:
                    inner_walls.append((start, end))
    
            if cell['W']:
                start = (x_pos, y_pos)
                end = (x_pos, y_pos + cell_size)
                if x == 0:
                    outer_walls.append((start, end))
                else:
                    inner_walls.append((start, end))
    
            if cell['E']:
                start = (x_pos + cell_size, y_pos)
                end = (x_pos + cell_size, y_pos + cell_size)
                if x == n - 1:
                    outer_walls.append((start, end))
                else:
                    inner_walls.append((start, end))
    
    glLineWidth(inner_line_width)
    glBegin(GL_LINES)
    for start, end in inner_walls:
        glVertex2f(*start)
        glVertex2f(*end)
    glEnd()
    
    glLineWidth(outer_line_width)
    glBegin(GL_LINES)
    for start, end in outer_walls:
        glVertex2f(*start)
        glVertex2f(*end)
    glEnd()
    
    glLineWidth(1.0)

def draw_cell_highlight(position, cell_size, margin, n):
    """Render a highlighted cell."""
    x, y = position
    x_pos = -1 + margin + x * cell_size
    y_pos = 1 - margin - (y + 1) * cell_size  # Invert y-axis
    glBegin(GL_QUADS)
    glVertex2f(x_pos, y_pos)
    glVertex2f(x_pos + cell_size, y_pos)
    glVertex2f(x_pos + cell_size, y_pos + cell_size)
    glVertex2f(x_pos, y_pos + cell_size)
    glEnd()

def main():
    if not glfw.init():
        return

    window = glfw.create_window(800, 800, "Maze Solver Visualization", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(-1.0, 1.0, -1.0, 1.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glClearColor(1.0, 1.0, 1.0, 1.0)

    n = 8  # Maze size
    maze, entrance, exit = generate_maze(n)
    entrance_pos, entrance_side = entrance
    exit_pos, exit_side = exit

    graph = maze_to_graph(maze)
    dijkstra_gen = dijkstra_generator(graph, entrance_pos, exit_pos)
    algorithm_state = next(dijkstra_gen)

    frame_delay = 1.00  # Seconds per frame

    while not glfw.window_should_close(window):
        start_time = time.time()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_maze(maze, entrance, exit, algorithm_state)
        glfw.swap_buffers(window)
        glfw.poll_events()

        try:
            algorithm_state = next(dijkstra_gen)
        except StopIteration:
            pass

        elapsed_time = time.time() - start_time
        if elapsed_time < frame_delay:
            time.sleep(frame_delay - elapsed_time)

    glfw.terminate()

if __name__ == "__main__":
    main()
