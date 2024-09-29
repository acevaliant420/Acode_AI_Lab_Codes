# 1.5
# Code to generate Puzzle-8 instances with the goal state at depth “d”

def generate_puzzle_instance(goal_state, depth):
    current_state = goal_state  # Start with the goal state

    # Apply random moves up to the given depth to generate the puzzle instance
    for _ in range(depth):
        # Get the possible successors of the current state
        successors = get_successors(Node(current_state))

        # Randomly choose one successor as the next state
        next_state = random.choice(successors)

        # Update the current state to the chosen successor
        current_state = next_state

    return current_state  # Return the generated puzzle instance
