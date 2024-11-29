import numpy as np
import math
from functools import partial
from multiprocessing import Pool

# Parameters
MAX_BIKES = 15  # Reduced for testing
MAX_MOVE = 3  # Reduced for faster computation
MOVE_COST = -2
ADDITIONAL_PARK_COST = -4
RENT_REWARD = 10
RENTAL_RATE_FIRST_LOC = 3
RENTAL_RATE_SECOND_LOC = 4
RETURN_RATE_FIRST_LOC = 3
RETURN_RATE_SECOND_LOC = 2
TRUNCATE = 6
DELTA = 1e-1
GAMMA = 0.9

# Precompute Poisson probabilities
poisson_cache = {}
def poisson_prob(x, lam):
    if (x, lam) not in poisson_cache:
        poisson_cache[(x, lam)] = np.exp(-lam) * pow(lam, x) / math.factorial(x)
    return poisson_cache[(x, lam)]

# States
states = [(i, j) for i in range(MAX_BIKES + 1) for j in range(MAX_BIKES + 1)]

class PolicyIteration:
    def __init__(self):
        self.actions = np.arange(-MAX_MOVE, MAX_MOVE + 1)
        self.values = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1))
        self.policy = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1), dtype=int)

    def solve(self):
        iterations = 0
        while True:
            self.values = self.evaluate_policy()
            policy_change = self.improve_policy()
            if policy_change == 0:
                break
            iterations += 1

    def evaluate_policy(self):
        values = self.values.copy()
        while True:
            new_values = np.zeros_like(values)
            with Pool() as pool:
                results = pool.map(partial(self._compute_state_value, values=values), states)
            for (value, (i, j)) in results:
                new_values[i, j] = value
            if np.abs(new_values - values).max() < DELTA:
                return new_values
            values = new_values

    def improve_policy(self):
        policy_change = 0
        for (i, j) in states:
            old_action = self.policy[i, j]
            action_returns = [self._compute_action_return((i, j), action, self.values) for action in self.actions]
            self.policy[i, j] = self.actions[np.argmax(action_returns)]
            if old_action != self.policy[i, j]:
                policy_change += 1
        return policy_change

    def _compute_state_value(self, state, values):
        i, j = state
        action = self.policy[i, j]
        return self._compute_action_return(state, action, values), state

    def _compute_action_return(self, state, action, values):
        i, j = state
        expected_return = MOVE_COST * abs(action)
        gbikes_first_loc = min(max(i - action, 0), MAX_BIKES)
        gbikes_second_loc = min(max(j + action, 0), MAX_BIKES)

        for req1 in range(TRUNCATE):
            for req2 in range(TRUNCATE):
                prob_rental = poisson_prob(req1, RENTAL_RATE_FIRST_LOC) * poisson_prob(req2, RENTAL_RATE_SECOND_LOC)
                real_rentals_first_loc = min(req1, gbikes_first_loc)
                real_rentals_second_loc = min(req2, gbikes_second_loc)
                reward = (real_rentals_first_loc + real_rentals_second_loc) * RENT_REWARD

                if gbikes_first_loc >= 10:
                    reward += ADDITIONAL_PARK_COST
                if gbikes_second_loc >= 10:
                    reward += ADDITIONAL_PARK_COST

                gbikes_first_loc_after_rentals = gbikes_first_loc - real_rentals_first_loc
                gbikes_second_loc_after_rentals = gbikes_second_loc - real_rentals_second_loc

                for ret1 in range(TRUNCATE):
                    for ret2 in range(TRUNCATE):
                        prob_return = poisson_prob(ret1, RETURN_RATE_FIRST_LOC) * poisson_prob(ret2, RETURN_RATE_SECOND_LOC)
                        final_first_loc = min(gbikes_first_loc_after_rentals + ret1, MAX_BIKES)
                        final_second_loc = min(gbikes_second_loc_after_rentals + ret2, MAX_BIKES)
                        prob = prob_rental * prob_return
                        expected_return += prob * (reward + GAMMA * values[final_first_loc, final_second_loc])
        return expected_return

    def plot_policy(self):
        import matplotlib.pyplot as plt
        plt.imshow(np.flipud(self.policy), cmap="coolwarm", interpolation="none")
        plt.colorbar(label="Policy (Action)")
        plt.xlabel("Bikes at Location 2")
        plt.ylabel("Bikes at Location 1")
        plt.title("Optimal Policy")
        plt.show()

    def plot_value(self):
        import matplotlib.pyplot as plt
        plt.imshow(np.flipud(self.values), cmap="viridis", interpolation="none")
        plt.colorbar(label="Value")
        plt.xlabel("Bikes at Location 2")
        plt.ylabel("Bikes at Location 1")
        plt.title("Value Function")
        plt.show()

if __name__ == "__main__":
    solver = PolicyIteration()
    solver.solve()
    solver.plot_policy()
    solver.plot_value()
