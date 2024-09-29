from collections import deque

def is_valid(state):
    missionaries, cannibals, boat = state
    if missionaries < 0 or cannibals < 0 or missionaries > 3 or cannibals > 3:
        return False
    if missionaries > 0 and missionaries < cannibals:
        return False
    if 3 - missionaries > 0 and 3 - missionaries < 3 - cannibals:
        return False
    return True

def get_successors(state):
    successors = []
    missionaries, cannibals, boat = state
    if boat == 1:
        moves = [(2, 0), (0, 2), (1, 1), (1, 0), (0, 1)]
        for move in moves:
            new_state = (missionaries - move[0], cannibals - move[1], 0)
            if is_valid(new_state):
                successors.append(new_state)
    else:
        moves = [(2, 0), (0, 2), (1, 1), (1, 0), (0, 1)]
        for move in moves:
            new_state = (missionaries + move[0], cannibals + move[1], 1)
            if is_valid(new_state):
                successors.append(new_state)
    return successors

def dfs(start_state, goal_state):
    stack = [(start_state, [])]
    visited = set()
    nodes_visited = 0  # Counter for visited nodes
    
    while stack:
        state, path = stack.pop()
        if state in visited:
            continue
        visited.add(state)
        nodes_visited += 1  # Increment counter for each new visited node
        path = path + [state]
        if state == goal_state:
            print(f"Total nodes visited (DFS): {nodes_visited}")
            return path
        for successor in get_successors(state):
            stack.append((successor, path))
    
    print(f"Total nodes visited (DFS): {nodes_visited}")
    return None

start_state = (3, 3, 1)
goal_state = (0, 0, 0)

solution = dfs(start_state, goal_state)
if solution:
    print("Following Traversal reaches Goal State:")
    for step in solution:
        print(step)
else:
    print("Can't reach goal state.")
