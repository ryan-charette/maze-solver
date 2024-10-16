import heapq

def maze_to_graph(maze):
    """Convert maze representation to adjacency list graph."""
    n = len(maze)
    graph = {}
    for y in range(n):
        for x in range(n):
            node = (x, y)
            graph[node] = []
            cell = maze[y][x]
            if not cell['N'] and y > 0:
                graph[node].append(((x, y - 1), 1))
            if not cell['S'] and y < n - 1:
                graph[node].append(((x, y + 1), 1))
            if not cell['E'] and x < n - 1:
                graph[node].append(((x + 1, y), 1))
            if not cell['W'] and x > 0:
                graph[node].append(((x - 1, y), 1))
    return graph

def dijkstra(graph, start, end):
    """Compute shortest path using Dijkstra's algorithm."""
    heap = [(0, start)]
    distances = {start: 0}
    previous_nodes = {start: None}
    visited = set()

    while heap:
        current_distance, current_node = heapq.heappop(heap)

        if current_node in visited:
            continue
        visited.add(current_node)

        if current_node == end:
            break

        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            if neighbor not in distances or distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(heap, (distance, neighbor))
                previous_nodes[neighbor] = current_node

    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous_nodes.get(current)
    path.reverse()
    return path

def dijkstra_generator(graph, start, end):
    """Yield states for animating Dijkstra's algorithm."""
    heap = [(0, start)]
    distances = {start: 0}
    previous_nodes = {start: None}
    visited = set()

    while heap:
        current_distance, current_node = heapq.heappop(heap)

        if current_node in visited:
            continue
        visited.add(current_node)

        yield {
            'current_node': current_node,
            'visited': visited.copy(),
            'distances': distances.copy(),
            'heap': heap.copy(),
            'previous_nodes': previous_nodes.copy(),
        }

        if current_node == end:
            break

        for neighbor, weight in graph[current_node]:
            if neighbor in visited:
                continue
            distance = current_distance + weight
            if neighbor not in distances or distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(heap, (distance, neighbor))
                previous_nodes[neighbor] = current_node

    path = []
    current = end if end in distances else None
    while current is not None:
        path.append(current)
        current = previous_nodes.get(current)
    path.reverse()

    yield {
        'current_node': None,
        'visited': visited.copy(),
        'distances': distances.copy(),
        'heap': heap.copy(),
        'previous_nodes': previous_nodes.copy(),
        'path': path,
    }
