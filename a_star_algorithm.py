import heapq
import math
import random
import matplotlib.pyplot as plt

def heuristic(node, goal):
    """
    Calculate the heuristic value from the current node to the goal.
    This can be Manhattan distance, Euclidean distance, or other based on the graph type.
    
    Parameters:
    node (tuple or object): Current node position.
    goal (tuple or object): Goal node position.
    
    Returns:
    float: The heuristic value.
    """
    # return abs(goal[0] - node[0]) + abs(goal[1] - node[1])  # Manhattan distance = |x(g)-x(c)| + |y(g)-y(c)|
    return math.sqrt(abs(goal[0] - node[0]) ** 2 + abs(goal[1] - node[1]) ** 2) # Euclidean distance = sqrt((x(g)-x(c))^2 + (y(g)-y(c))^2)

def a_star(graph, start, goal):
    """
    A* algorithm to find the shortest path from start to goal in a graph.

    Parameters:
    graph (dict): Graph with nodes as keys and lists of (neighbor_x, neighbor_y, cost) as values.
    start (tuple): Starting node.
    goal (tuple): Goal node.

    Returns:
    list: Shortest path from start to goal.
    float: Total cost of the path.
    """

    # Dictionary to reconstruct the path after reaching the goal
    came_from = {}
    
    # Cost from start to each node (g_cost) and total cost (f_cost)
    g_cost = {start: 0}
    f_cost = {start: heuristic(start, goal)}

    # Priority queue to store nodes to explore
    open_set = [(f_cost[start], start)]
    closed_set = set()

    while open_set:
        # Get the node with the lowest F-cost
        _, current = heapq.heappop(open_set)

        # Add node to closed_set
        closed_set.add(current)

        # If the goal node is reached, reconstruct the path and return
        if current == goal: return reconstruct_path(came_from, current), g_cost[current]

        # For each neighbor of the current node
        for neighbor_x, neighbor_y, h_cost in graph[current]:
            neighbor = (neighbor_x, neighbor_y)

            # Skip if the neighbor is in the closed set
            if neighbor in closed_set: continue

            # Calculate the new G-cost for the neighbor
            cost_to_neighbor = g_cost[current] + h_cost

            # If neighbor is not in the open set or has a lower G-cost, update costs
            if neighbor not in g_cost or cost_to_neighbor < g_cost[neighbor]:
                g_cost[neighbor] = cost_to_neighbor
                f_cost[neighbor] = cost_to_neighbor + heuristic(neighbor, goal)
                came_from[neighbor] = current

                # If neighbor is not in the open set, add it with the updated F-cost
                if neighbor not in [node for _, node in open_set]: heapq.heappush(open_set, (f_cost[neighbor], neighbor))

    print("No path exists.")
    exit(1)


def reconstruct_path(came_from, current):
    """
    Reconstruct the path from the start to the goal by backtracking from the goal node.
    
    Parameters:
    came_from (dict): Dictionary mapping nodes to their predecessors.
    current (tuple or object): The current node to backtrack from.
    
    Returns:
    list: The reconstructed path.
    """
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]

def test_a_star(test_cases):
    for index, case in enumerate(test_cases, start=1):
        maze = create_maze(case["size"][0], case["size"][1], obstacle_percentage=case["obstacle_percentage"], weighted_percentage=case["weighted_percentage"])
        start, goal = case["start"], case["goal"]
        graph = create_graph(maze)
        path, cost = a_star(graph, start, goal)

        print(f"Test Case {index}: Path = {path}, Cost = {cost}")

        #  print_maze and visualize_path are mirrored on y-axis because I inverted y-axis in visualize_path to make it look better
        #  print_maze(maze)
        #  print_graph(graph)

        visualize_path(f"Test Case {index}", maze, path, cost, start, goal)

def visualize_path(figName, maze, path, cost, start, goal):
    # Create figure with specified window name and figure size
    plt.figure(num=figName, figsize=(7, 7))
    
    # Set plot title to show the start and goal coordinates and the cost of the solution path
    plt.title(f"{figName} - Start: {start}, Goal: {goal}, Cost = {cost}")
    
    # Create list to represent the maze in a form readable by matplotlib
    maze_list = [[1 if cell == '#' else 0 for cell in row] for row in maze]

    plt.imshow(maze_list, cmap='magma_r')

    # Invert Y-Axis to make (0, 0) in the bottom-left instead of top-left
    # (This makes the graph mirrored on y-axis compared to print_maze function)
    plt.gca().invert_yaxis()

    # Mark start and goal points and create legend
    plt.scatter(*start[::-1], color="green", label="Start", s=25)
    plt.scatter(*goal[::-1], color="red", label="Goal", s=25)
    plt.legend(loc='upper left')
    
    # on_close is used here to detect when the user tries to exit the window of the figure
    # and allows the script to skip the graphing process if the user exists the
    # window instead of bugging and showing black window till animation is over
    exited = False
    def on_close(event):
        nonlocal exited
        exited = True
    plt.gcf().canvas.mpl_connect('close_event', on_close)
    
    # Get the color for the path
    path_color = plt.get_cmap('magma_r')(0.5)

    # Get line width relative to maze size
    linewidth = max(1, int(150 / len(maze_list)))

    # Animate the path cell by cell
    for i in range(1, len(path)):
        # Exit out of animation if user tried to exit the window of the figure
        if exited: break
        
        # Get current and previous point
        (x_prev, y_prev), (x, y) = path[i - 1], path[i]

        # Plot line between cells with the color specified earlier
        plt.plot([x_prev, x], [y_prev, y], color=path_color, linewidth=linewidth)
        plt.draw()
        plt.pause(0.05)

    plt.show()

def create_maze(width, height, obstacle_percentage=0.2, weighted_percentage=0.1, seed=42):
    """
    Create a 50x50 maze with random obstacles and weighted paths, but the same maze will
    be generated each time because of a fixed random seed.
    
    Parameters:
    width (int): Width of the maze.
    height (int): Height of the maze.
    obstacle_percentage (float): Percentage of cells that are obstacles.
    weighted_percentage (float): Percentage of cells that have a higher traversal cost.
    seed (int): Random seed to ensure the maze is the same each time.
    
    Returns:
    list: A 2D grid representing the maze. 
          '1' represents normal paths, 
          '#' represents obstacles, 
          values 2-5 represent weighted paths.
    """
    # Set the random seed to ensure consistent maze generation
    random.seed(seed)
    
    maze = []
    
    for _ in range(height):
        row = []
        for _ in range(width):
            # Randomly decide if the cell is an obstacle
            if random.random() < obstacle_percentage:
                row.append('#')  # Obstacle
            else:
                # Randomly assign weighted paths
                if random.random() < weighted_percentage:
                    row.append(random.randint(2, 5))  # Weighted path
                else:
                    row.append(1)  # Normal path with cost 1
        maze.append(row)
    
    return maze

def print_maze(maze):
    """
    Prints the maze in a human-readable format.
    
    Parameters:
    maze (list): A 2D grid representing the maze.
    """
    for row in maze:
        print(' '.join(str(cell) for cell in row))

def get_neighbors(x, y, maze):
    """
    Get valid neighbors of a cell (x, y) in the maze.
    
    Parameters:
    x (int): X-coordinate of the cell.
    y (int): Y-coordinate of the cell.
    maze (list): The maze represented as a 2D grid.
    
    Returns:
    list: List of valid neighbors as (neighbor_x, neighbor_y, cost) tuples.
    """
    neighbors = []
    height = len(maze)
    width = len(maze[0])
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
    
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height and maze[ny][nx] != '#':
            neighbors.append((nx, ny, maze[ny][nx]))
    
    return neighbors

def create_graph(maze):
    """
    Create a graph from the maze where each cell is a key, and its neighbors (with weights) are the values.
    
    Parameters:
    maze (list): The maze represented as a 2D grid.
    
    Returns:
    dict: A dictionary representing the graph, where each key is a (x, y) tuple and the value is
          a list of neighbors with their respective costs [(neighbor_x, neighbor_y, cost), ...].
    """
    graph = {}
    
    for y in range(len(maze)):
        for x in range(len(maze[0])):
            if maze[y][x] != '#':  # If the cell is not an obstacle
                graph[(x, y)] = get_neighbors(x, y, maze)
    
    return graph

def print_graph(graph):
    """
    Print the graph structure where each cell has a list of its neighbors with their weights.
    
    Parameters:
    graph (dict): The graph structure.
    """
    for node, neighbors in graph.items():
        print(f"Cell {node}: {neighbors}")

# Set dark mode for matplotlib
plt.style.use('dark_background')

# Hide the toolbar
plt.rcParams['toolbar'] = 'None'

# Example test cases
test_cases = [
        {"size": (50, 50), "start": (0, 0), "goal": (49, 49), "obstacle_percentage": 0.2, "weighted_percentage": 0.1},
        {"size": (50, 50), "start": (0, 0), "goal": (49, 49), "obstacle_percentage": 0.1, "weighted_percentage": 0.1},
        {"size": (50, 50), "start": (0, 0), "goal": (49, 49), "obstacle_percentage": 0.2, "weighted_percentage": 0.3},
        {"size": (75, 75), "start": (0, 0), "goal": (74, 74), "obstacle_percentage": 0.2, "weighted_percentage": 0.5},
        {"size": (75, 75), "start": (0, 0), "goal": (74, 74), "obstacle_percentage": 0.1, "weighted_percentage": 0.3},
        {"size": (25, 25), "start": (0, 0), "goal": (24, 24), "obstacle_percentage": 0.2, "weighted_percentage": 0.3}
    ]

test_a_star(test_cases)
