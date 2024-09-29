from collections import deque

# Check if the move is valid
def is_valid(state):
    return -1 <= state.index(0) <= 6  # Ensures the empty space is in a valid position

# Generate successor states
def get_successors(state):
    successors = []
    empty_index = state.index(0)
    
    # Possible moves: 1 step or 2 steps in either direction
    moves = [-1, -2, 1, 2]
    
    for move in moves:
        new_index = empty_index + move
        if 0 <= new_index <= 6:
            # Swap the empty space with the rabbit in the new position
            new_state = list(state)
            new_state[empty_index], new_state[new_index] = new_state[new_index], new_state[empty_index]
            successors.append(tuple(new_state))
    
    return successors

# DFS implementation


def dfs(start_state, goal_state):
    stack = [(start_state, [])]
    visited = set()
    nodes_visited = 0
    
    while stack:
        state, path = stack.pop()
        if state in visited:
            continue
        visited.add(state)
        nodes_visited += 1
        path = path + [state]
        if state == goal_state:
            print(f"Total nodes visited (DFS): {nodes_visited}")
            return path
        for successor in get_successors(state):
            stack.append((successor, path))
    
    print(f"Total nodes visited (DFS): {nodes_visited}")
    return None
start_state = (-1, -1, -1, 0, 1, 1, 1)
goal_state = (1, 1, 1, 0, -1, -1, -1)
solution = dfs(start_state, goal_state)
if solution:
    print("Following Traversal reaches Goal State:")
    for step in solution:
        print(step)
else:
    print("Can't reach goal state.")
