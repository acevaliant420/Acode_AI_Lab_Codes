import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

MAX_GBIKE = 20
MAX_MOVE_OF_GBIKE = 5
RENTAL_REQUEST_FIRST_LOC = 3
RENTAL_REQUEST_SECOND_LOC = 4
RETURNS_FIRST_LOC = 3
RETURNS_SECOND_LOC = 2
GAMMA = 0.9
RENTAL_CREDIT = 10
MOVE_GBIKE_COST = 2
POISSON_UPPER_BOUND = 15  

actions = np.arange(-MAX_MOVE_OF_GBIKE, MAX_MOVE_OF_GBIKE + 1)
value = np.zeros((MAX_GBIKE + 1, MAX_GBIKE + 1))
policy = np.zeros(value.shape, dtype=np.int64)

poisson_cache = dict()


def poisson_dist(x, lam):
    
    key = (x, lam)
    if key not in poisson_cache:
        poisson_cache[key] = poisson.pmf(x, lam)
    return poisson_cache[key]


def expected_return(state, action, state_value):
    
    returns = -MOVE_GBIKE_COST * abs(action)

    num_gbike_first_loc = int(max(min(state[0] - action, MAX_GBIKE), 0))
    num_gbike_second_loc = int(max(min(state[1] + action, MAX_GBIKE), 0))

    for rental_request_first_loc in range(POISSON_UPPER_BOUND):
        for rental_request_second_loc in range(POISSON_UPPER_BOUND):
            prob = poisson_dist(rental_request_first_loc, RENTAL_REQUEST_FIRST_LOC) * \
                   poisson_dist(rental_request_second_loc, RENTAL_REQUEST_SECOND_LOC)

            valid_rentals_first_loc = min(num_gbike_first_loc, rental_request_first_loc)
            valid_rentals_second_loc = min(num_gbike_second_loc, rental_request_second_loc)

            reward = (valid_rentals_first_loc + valid_rentals_second_loc) * RENTAL_CREDIT

            temp_gbike_first_loc = num_gbike_first_loc - valid_rentals_first_loc
            temp_gbike_second_loc = num_gbike_second_loc - valid_rentals_second_loc

            temp_gbike_first_loc = min(temp_gbike_first_loc + RETURNS_FIRST_LOC, MAX_GBIKE)
            temp_gbike_second_loc = min(temp_gbike_second_loc + RETURNS_SECOND_LOC, MAX_GBIKE)


            returns += prob * (reward + GAMMA * state_value[temp_gbike_first_loc, temp_gbike_second_loc])

    return returns


def policy_evaluation():
    global value
    while True:
        old_value = value.copy()
        for i in range(MAX_GBIKE + 1):
            for j in range(MAX_GBIKE + 1):
                value[i, j] = expected_return([i, j], policy[i, j], value)
        max_value_change = abs(old_value - value).max()
        print(f'Max value change: {max_value_change}')
        if max_value_change < 1e-4:
            break


def policy_improvement():
    global policy
    policy_stable = True
    for i in range(MAX_GBIKE + 1):
        for j in range(MAX_GBIKE + 1):
            old_action = policy[i, j]
            action_returns = []
            for action in actions:
                if -j <= action <= i:  
                    action_returns.append(expected_return([i, j], action, value))
                else:
                    action_returns.append(-np.inf)
            policy[i, j] = actions[np.argmax(action_returns)]
            if old_action != policy[i, j]:
                policy_stable = False
    print(f'Policy stable: {policy_stable}')
    return policy_stable


def main():
    iterations = 0
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    axes = axes.flatten()

    while True:
        if iterations < len(axes) - 1:
            ax = axes[iterations]
            im = ax.imshow(np.flipud(policy), cmap='YlGnBu', aspect='auto')
            ax.set_title(f'Policy {iterations}', fontsize=16)
            ax.set_xlabel('# of bikes at second location', fontsize=12)
            ax.set_ylabel('# of bikes at first location', fontsize=12)
            ax.set_xticks(range(0, MAX_GBIKE + 1, 5))
            ax.set_yticks(range(0, MAX_GBIKE + 1, 5))
            ax.set_yticklabels(reversed(range(0, MAX_GBIKE + 1, 5)))

        policy_evaluation()

        policy_stable = policy_improvement()

        if policy_stable:
            ax = axes[-1]
            im = ax.imshow(np.flipud(value), cmap='YlGnBu', aspect='auto')
            ax.set_title('Optimal Value Function', fontsize=16)
            ax.set_xlabel('# of bikes at second location', fontsize=12)
            ax.set_ylabel('# of bikes at first location', fontsize=12)
            ax.set_xticks(range(0, MAX_GBIKE + 1, 5))
            ax.set_yticks(range(0, MAX_GBIKE + 1, 5))
            ax.set_yticklabels(reversed(range(0, MAX_GBIKE + 1, 5)))
            fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.6)
            break

        iterations += 1

    plt.tight_layout()
    plt.savefig('figure_gbike_matplotlib.png')
    plt.show()


if __name__ == '__main__':
    main()
