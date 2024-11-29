import numpy as np

TERMINAL_STATES = {(1, 3), (2, 3)}
WALLS = {(1, 1)}
POSSIBLE_ACTIONS = ['L', 'R', 'U', 'D']
ACTION_PROBABILITIES = {'L': 0.25, 'R': 0.25, 'U': 0.25, 'D': 0.25}
ENV_TRANSITIONS_LEFT = {'L': 'D', 'R': 'U', 'U': 'L', 'D': 'R'}
ENV_TRANSITIONS_RIGHT = {'L': 'U', 'R': 'D', 'U': 'R', 'D': 'L'}

def initialize_reward_matrix(r):
    reward_matrix = [[r for _ in range(4)] for _ in range(3)]
    reward_matrix[2][3] = 1
    reward_matrix[1][3] = -1
    return reward_matrix

def is_valid_state(x, y):
    return (x, y) not in WALLS and 0 <= x < 3 and 0 <= y < 4

def display_values(values):
    for i in range(2, -1, -1):
        print("---------------------------")
        for j in range(4):
            v = values[i][j]
            print(f"{v:6.2f}|", end="")
        print("")

def move(action, x, y):
    if action == 'L':
        return x, y - 1
    elif action == 'R':
        return x, y + 1
    elif action == 'U':
        return x + 1, y
    elif action == 'D':
        return x - 1, y

def compute_value(x, y, V, reward_matrix, gamma=1):
    value = 0
    for action in POSSIBLE_ACTIONS:
        new_x, new_y = move(action, x, y)
        value_given = reward_matrix[new_x][new_y] + gamma * V[new_x][new_y] if is_valid_state(new_x, new_y) else reward_matrix[x][y] + gamma * V[x][y]

        left_x, left_y = move(ENV_TRANSITIONS_LEFT[action], x, y)
        value_left = reward_matrix[left_x][left_y] + gamma * V[left_x][left_y] if is_valid_state(left_x, left_y) else reward_matrix[x][y] + gamma * V[x][y]

        right_x, right_y = move(ENV_TRANSITIONS_RIGHT[action], x, y)
        value_right = reward_matrix[right_x][right_y] + gamma * V[right_x][right_y] if is_valid_state(right_x, right_y) else reward_matrix[x][y] + gamma * V[x][y]

        value_action = 0.8 * value_given + 0.1 * value_left + 0.1 * value_right
        value += value_action * ACTION_PROBABILITIES[action]
    return value

def value_iteration(r, gamma=1, threshold=1e-6):
    reward_matrix = initialize_reward_matrix(r)
    V = [[0 for _ in range(4)] for _ in range(3)]
    iterations = 0

    while True:
        delta = 0
        for i in range(3):
            for j in range(4):
                if (i, j) in TERMINAL_STATES or (i, j) in WALLS:
                    continue
                v_old = V[i][j]
                V[i][j] = compute_value(i, j, V, reward_matrix, gamma)
                delta = max(delta, abs(v_old - V[i][j]))
        iterations += 1
        if delta < threshold:
            break
    return V, iterations

reward_values = [-2, 0.1, 0.02, 1]
for r in reward_values:
    print(f"\nValue Function for r(s) = {r}:")
    V, iterations = value_iteration(r)
    display_values(V)
    print(f"Converged in {iterations} iterations.")
