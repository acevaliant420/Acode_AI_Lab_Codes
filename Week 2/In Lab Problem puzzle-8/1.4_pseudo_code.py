#1.4
# function which can backtrack and produce the path taken to reach the goal state from the source/ initial state
def backtrack_path(goal_node):
    path = []  # Initialize an empty list to store the path
    current_node = goal_node  # Start backtracking from the goal node

    # Traverse up the tree by following parent links
    while current_node is not None:
        path.insert(0, current_node.state)  # Prepend the current state to the path
        current_node = current_node.parent  # Move to the parent node

    return path  # Return the final solution path from source to goal