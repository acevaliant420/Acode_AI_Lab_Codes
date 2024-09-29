import heapq
import random

class Node:
    def __init__(self, state, parent=None, g=0, h=0):
        self.state = state
        self.parent = parent
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f

def heuristic(state, goal_state):
    h = 0
    for i in range(9):
        if state[i] != 0 and state[i] != goal_state[i]:
            h += 1
    return h

def get_successors(node):
    successors = []
    index = node.state.index(0)
    quotient = index // 3
    remainder = index % 3

    if quotient == 0:
        moves = [3]
    elif quotient == 1:
        moves = [-3, 3]
    elif quotient == 2:
        moves = [-3]

    if remainder == 0:
        moves += [1]
    elif remainder == 1:
        moves += [-1, 1]
    elif remainder == 2:
        moves += [-1]

    for move in moves:
        im = index + move
        if 0 <= im < 9:
            new_state = list(node.state)
            new_state[index], new_state[im] = new_state[im], new_state[index]
            successors.append(new_state)
    return successors

def search_agent(start_state, goal_state):
    start_node = Node(start_state, None, 0, heuristic(start_state, goal_state))
    frontier = [(start_node.f, start_node)]
    visited = set()
    nodes_explored = 0

    while frontier:
        _, node = heapq.heappop(frontier)
        nodes_explored += 1

        if tuple(node.state) in visited:
            continue
        visited.add(tuple(node.state))

        if node.state == goal_state:
            path = []
            while node:
                path.append(node.state)
                node = node.parent
            print('Total nodes explored', nodes_explored)
            return path[::-1]

        for successor_state in get_successors(node):
            if tuple(successor_state) not in visited:
                g = node.g + 1
                h = heuristic(successor_state, goal_state)
                successor_node = Node(successor_state, node, g, h)
                heapq.heappush(frontier, (successor_node.f, successor_node))

    print('Total nodes explored', nodes_explored)
    return None

def generate_goal_state(start_state, depth):
    current_state = start_state.copy()
    for _ in range(depth):
        successors = get_successors(Node(current_state))
        current_state = random.choice(successors)
    return current_state

start_state = [1, 4, 2, 0, 6, 5, 3, 8, 7]
D = 20
goal_state = generate_goal_state(start_state, D)
print("Start state:", start_state)
print("Goal state:", goal_state)

solution = search_agent(start_state, goal_state)
if solution:
    print("Solution found:")
    for i, step in enumerate(solution):
        print(f"Step {i}:")
        for j in range(0, 9, 3):
            print(step[j:j+3])
        print()
else:
    print("No solution found.")
