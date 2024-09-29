# 1.1
# pseudocode for a graph search agent

def graph_search_agent(start_state, goal_state):
    # Initialize the priority queue (min-heap) for the frontier
    frontier = priority_queue()  # Min-heap based on node's f value (g + h)
    explored_set = set()  # Set to track visited states
    nodes_explored = 0  # Counter for explored nodes

    # Create the start node with g = 0 and heuristic h
    start_node = Node(start_state, g=0, h=heuristic(start_state, goal_state))
    frontier.push(start_node)

    # Loop until the frontier is empty
    while frontier is not empty:
        # Pop the node with the lowest f value
        node = frontier.pop()
        nodes_explored += 1

        # Check if the goal state is reached
        if node.state == goal_state:
            return backtrack_path(node)  # Return the solution path

        # Add the current node's state to the explored set
        explored_set.add(node.state)

        # Generate all successor states of the current node
        for successor_state in get_successors(node):
            # Only explore new states (not in explored_set)
            if successor_state not in explored_set:
                # Calculate new g (cost) and h (heuristic) values
                g = node.g + 1
                h = heuristic(successor_state, goal_state)
                successor_node = Node(successor_state, parent=node, g=g, h=h)

                # Push the successor node into the frontier
                frontier.push(successor_node)

    return None  # Return failure if no solution is found