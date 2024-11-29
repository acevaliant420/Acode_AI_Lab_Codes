import numpy as np
import random
def generate_random_initial_state(size, num_rooks=8):

    board = np.zeros((size, size), dtype=int) 
    rook_positions = np.random.choice(size * size, num_rooks, replace=False)  
    for pos in rook_positions:
        row, col = divmod(pos, size)
        board[row, col] = 1  
    return board.flatten() 
    

def initialize_weights(size):

    weights = np.zeros((size * size, size * size))

    for i in range(size):
        for j in range(size):
            for k in range(size):
                for l in range(size):
                    if i == k and j != l:
                        weights[i * size + j, k * size + l] = -1
                    if j == l and i != k:
                        weights[i * size + j, k * size + l] = -1
    return weights

def energy(state, weights):
    return -0.5 * np.dot(state.T, np.dot(weights, state))

def hopfield_dynamics(state, weights, threshold=0):

    size = len(state)
    updated_state = state.copy()
    for i in range(size):
        net_input = np.dot(weights[i], updated_state) 
        updated_state[i] = 1 if net_input > threshold else 0  
    return updated_state

def eight_rook_hopfield(size=8, num_iterations=100, threshold=0):

    state = generate_random_initial_state(size)
    print("Initial State (Flattened):")
    print(state.reshape(size, size))

    weights = initialize_weights(size)

    for iteration in range(num_iterations):
        new_state = hopfield_dynamics(state, weights, threshold)
        
        if np.array_equal(state, new_state):
            print(f"Converged after {iteration + 1} iterations.")
            break
        state = new_state

    print("Final State (Flattened):")
    print(state.reshape(size, size))
    return state.reshape(size, size)


np.random.seed(random.randint(1 , 40))  
solution = eight_rook_hopfield()
