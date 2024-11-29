import numpy as np



"""
    -----------------------------------------------------------
    Statistical Physics Approach (Storkey Heuristic):

        -> Capacity ≈ 0.138 * N
        -> N = 100 neurons
        -> Estimated Capacity ≈ 14 distinct patterns

    ------------------------------------------------------------
    Classic Gardner Bound:

        -> Capacity ≈ 0.15 * N
        -> Capacity ≈ 15 distinct patterns

    ------------------------------------------------------------
    Computational Learning Theory Estimate:

        -> Roughly 0.12 to 0.14 * N
        -> Capacity around 12-14 patterns

    ------------------------------------------------------------
    """
class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)

    def recall(self, pattern, steps=5):
        for _ in range(steps):
            new_pattern = pattern[:]
            for i in range(self.size):
                activation = sum(self.weights[i][j] * pattern[j] for j in range(self.size))
                new_pattern[i] = 1 if activation >= 0 else -1
            pattern = new_pattern
        return pattern

    def print_pattern(self, pattern):
        n = int(np.sqrt(self.size))
        pattern_reshaped = np.array(pattern).reshape(n, n)
        print("Pattern:")
        for row in pattern_reshaped:
            print(' '.join(['*' if x == 1 else ' ' for x in row]))


pattern1 = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1] * 10)
pattern2 = np.array([-1, 1, -1, 1, -1, 1, -1, 1, -1, 1] * 10)
patterns = [pattern1, pattern2]

hopefield = HopfieldNetwork(size=100)

hopefield.train(patterns)

test_pattern = pattern1.copy()
np.random.seed(42)
noise_indices = np.random.choice(100, 10, replace=False)
for idx in noise_indices:
    test_pattern[idx] = -test_pattern[idx]

print("Test pattern (with noise):")
hopefield.print_pattern(test_pattern)

recalled_pattern = hopefield.recall(test_pattern)

print("\nRecalled pattern (after recall process):")
hopefield.print_pattern(recalled_pattern)
