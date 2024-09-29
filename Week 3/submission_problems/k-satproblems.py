import random

def create_3_sat_instance(num_vars, num_clauses):
    clauses = []
    for _ in range(num_clauses):
        clause = set()
        while len(clause) < 3:
            var = random.randint(1, num_vars)
            literal = var if random.choice([True, False]) else -var
            clause.add(literal)
        clauses.append(list(clause))
    return clauses

num_vars = 5
num_clauses = 10
three_sat_instance = create_3_sat_instance(num_vars, num_clauses)

print("Generated 3-SAT Instance:")
for clause in three_sat_instance:
    print(clause)
