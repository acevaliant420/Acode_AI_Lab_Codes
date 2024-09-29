import random
def generate_k_sat(n_variables, n_clauses, k=3):
    clauses = []
    for _ in range(n_clauses):
        clause = []
        while len(clause) < k:
            var = random.randint(1, n_variables)
            if var not in clause and -var not in clause:
                sign = random.choice([True, False])
                literal = var if sign else -var
                clause.append(literal)
        clauses.append(clause)
    return clauses


def evaluate_k_sat(clauses, assignment):
    for clause in clauses:
        if not any(
                (literal > 0 and assignment[abs(literal) - 1]) or (literal < 0 and not assignment[abs(literal) - 1]) for
                literal in clause):
            return False
    return True


def count_satisfied_clauses(clauses, assignment):
    return sum(1 for clause in clauses if any(
        (literal > 0 and assignment[abs(literal) - 1]) or (literal < 0 and not assignment[abs(literal) - 1]) for literal
        in clause))


def hill_climbing(clauses, n_variables, heuristic):
    assignment = [random.choice([True, False]) for _ in range(n_variables)]
    max_iters = 1000

    for _ in range(max_iters):
        if evaluate_k_sat(clauses, assignment):
            return assignment

        best_assignment = assignment.copy()
        best_score = heuristic(clauses, best_assignment)

        for i in range(n_variables):
            new_assignment = assignment.copy()
            new_assignment[i] = not new_assignment[i]
            new_score = heuristic(clauses, new_assignment)

            if new_score > best_score:
                best_assignment = new_assignment
                best_score = new_score

        assignment = best_assignment

    return None


def beam_search(clauses, n_variables, beam_width, heuristic):
    beam = [[random.choice([True, False]) for _ in range(n_variables)] for _ in range(beam_width)]
    max_iters = 1000

    for _ in range(max_iters):
        new_beam = []

        for assignment in beam:
            if evaluate_k_sat(clauses, assignment):
                return assignment
            for i in range(n_variables):
                new_assignment = assignment.copy()
                new_assignment[i] = not new_assignment[i]
                new_beam.append((new_assignment, heuristic(clauses, new_assignment)))

        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = [assignment for assignment, _ in new_beam[:beam_width]]

    return None


def vnd(clauses, n_variables, heuristic):
    assignment = [random.choice([True, False]) for _ in range(n_variables)]
    neighborhoods = [1, 2, 3]
    max_iters = 1000

    for _ in range(max_iters):
        if evaluate_k_sat(clauses, assignment):
            return assignment

        best_assignment = assignment.copy()
        best_score = heuristic(clauses, assignment)

        for neighborhood in neighborhoods:
            for i in range(n_variables - neighborhood + 1):
                new_assignment = assignment.copy()
                for j in range(neighborhood):
                    new_assignment[i + j] = not new_assignment[i + j]

                new_score = heuristic(clauses, new_assignment)
                if new_score > best_score:
                    best_assignment = new_assignment
                    best_score = new_score

        assignment = best_assignment

    return None


def heuristic_1(clauses, assignment):
    return count_satisfied_clauses(clauses, assignment)


def heuristic_2(clauses, assignment):
    return sum(1 for clause in clauses if len([literal for literal in clause if
                                               (literal > 0 and assignment[abs(literal) - 1]) or (
                                                           literal < 0 and not assignment[abs(literal) - 1])]) > 1)


def compare_algorithms(m_values, n_values):
    for m, n in zip(m_values, n_values):
        print(f"Solving 3-SAT problem with {m} clauses and {n} variables")
        clauses = generate_k_sat(n, m)

        hill_climbing_result_1 = hill_climbing(clauses, n, heuristic_1)
        hill_climbing_result_2 = hill_climbing(clauses, n, heuristic_2)
        print(f"Hill-Climbing result (heuristic 1): {'Satisfiable' if hill_climbing_result_1 else 'Unsatisfiable'}")
        print(f"Hill-Climbing result (heuristic 2): {'Satisfiable' if hill_climbing_result_2 else 'Unsatisfiable'}")

        beam_search_result_3_1 = beam_search(clauses, n, 3, heuristic_1)
        beam_search_result_3_2 = beam_search(clauses, n, 3, heuristic_2)
        print(f"Beam Search (width=3, heuristic 1): {'Satisfiable' if beam_search_result_3_1 else 'Unsatisfiable'}")
        print(f"Beam Search (width=3, heuristic 2): {'Satisfiable' if beam_search_result_3_2 else 'Unsatisfiable'}")

        beam_search_result_4_1 = beam_search(clauses, n, 4, heuristic_1)
        beam_search_result_4_2 = beam_search(clauses, n, 4, heuristic_2)
        print(f"Beam Search (width=4, heuristic 1): {'Satisfiable' if beam_search_result_4_1 else 'Unsatisfiable'}")
        print(f"Beam Search (width=4, heuristic 2): {'Satisfiable' if beam_search_result_4_2 else 'Unsatisfiable'}")

        vnd_result_1 = vnd(clauses, n, heuristic_1)
        vnd_result_2 = vnd(clauses, n, heuristic_2)
        print(f"Variable-Neighborhood-Descent (heuristic 1): {'Satisfiable' if vnd_result_1 else 'Unsatisfiable'}")
        print(f"Variable-Neighborhood-Descent (heuristic 2): {'Satisfiable' if vnd_result_2 else 'Unsatisfiable'}")


m_values = [10, 15, 20]
n_values = [5, 7, 10]
compare_algorithms(m_values, n_values)
