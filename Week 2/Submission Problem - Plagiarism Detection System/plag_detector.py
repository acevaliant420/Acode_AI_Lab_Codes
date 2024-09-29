import heapq
import re

def preprocess_text(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [sentence.lower().strip() for sentence in sentences]

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def calculate_similarity(s1, s2):
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return (1 - (distance / max_len)) * 100

class State:
    def __init__(self, pos1, pos2, cost, path):
        self.pos1 = pos1
        self.pos2 = pos2
        self.cost = cost
        self.path = path

    def __lt__(self, other):
        return self.cost < other.cost

def heuristic(state, doc1, doc2):
    remaining_sentences1 = len(doc1) - state.pos1
    remaining_sentences2 = len(doc2) - state.pos2
    return min(remaining_sentences1, remaining_sentences2)

def a_star_alignment(doc1, doc2):
    initial_state = State(0, 0, 0, [])
    heap = [(0, initial_state)]
    visited = set()

    while heap:
        _, current_state = heapq.heappop(heap)

        if current_state.pos1 == len(doc1) and current_state.pos2 == len(doc2):
            return current_state.path

        if (current_state.pos1, current_state.pos2) in visited:
            continue

        visited.add((current_state.pos1, current_state.pos2))

        if current_state.pos1 < len(doc1) and current_state.pos2 < len(doc2):
            cost = levenshtein_distance(doc1[current_state.pos1], doc2[current_state.pos2])
            new_state = State(current_state.pos1 + 1, current_state.pos2 + 1, current_state.cost + cost, current_state.path + [(current_state.pos1, current_state.pos2, cost)])
            heapq.heappush(heap, (new_state.cost + heuristic(new_state, doc1, doc2), new_state))

        if current_state.pos1 < len(doc1):
            new_state = State(current_state.pos1 + 1, current_state.pos2, current_state.cost + 1, current_state.path + [(current_state.pos1, None, 1)])
            heapq.heappush(heap, (new_state.cost + heuristic(new_state, doc1, doc2), new_state))

        if current_state.pos2 < len(doc2):
            new_state = State(current_state.pos1, current_state.pos2 + 1, current_state.cost + 1, current_state.path + [(None, current_state.pos2, 1)])
            heapq.heappush(heap, (new_state.cost + heuristic(new_state, doc1, doc2), new_state))

def find_best_pairs(doc1, doc2):
    pairs = []
    used_indices_doc1 = set()
    used_indices_doc2 = set()

    for i, sent1 in enumerate(doc1):
        for j, sent2 in enumerate(doc2):
            if i not in used_indices_doc1 and j not in used_indices_doc2:
                if sent1 == sent2:
                    pairs.append((i, j, 100.0))
                    used_indices_doc1.add(i)
                    used_indices_doc2.add(j)

    for i, sent1 in enumerate(doc1):
        if i not in used_indices_doc1:
            best_match = None
            best_similarity = -1
            best_j = -1
            for j, sent2 in enumerate(doc2):
                if j not in used_indices_doc2:
                    similarity = calculate_similarity(sent1, sent2)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = sent2
                        best_j = j
            if best_match is not None:
                pairs.append((i, best_j, best_similarity))
                used_indices_doc1.add(i)
                used_indices_doc2.add(best_j)

    return pairs

def detect_plagiarism(doc1, doc2, threshold=0.8):
    preprocessed_doc1 = preprocess_text(doc1)
    preprocessed_doc2 = preprocess_text(doc2)

    alignment = a_star_alignment(preprocessed_doc1, preprocessed_doc2)
    plagiarism_cases = []

    for i, j, cost in alignment:
        if i is not None and j is not None:
            max_length = max(len(preprocessed_doc1[i]), len(preprocessed_doc2[j]))
            similarity = 1 - (cost / max_length)
            if similarity >= threshold:
                plagiarism_cases.append((i, j, similarity * 100))

    best_pairs = find_best_pairs(preprocessed_doc1, preprocessed_doc2)
    total_similarity = sum(similarity for _, _, similarity in best_pairs)
    overall_percentage = total_similarity / len(preprocessed_doc1)

    return plagiarism_cases, overall_percentage

def plagiarism_verdict(percentage):
    if percentage > 80:
        return "Surely copied."
    elif percentage > 60:
        return "Likely copied."
    elif percentage > 40:
        return "Possibly copied."
    elif percentage > 20:
        return "Unlikely copied."
    else:
        return "Not copied."

def run_test_cases():
    test_cases = [
        ("Identical Documents",
         "This is a test. It has multiple sentences. We want to detect plagiarism.",
         "This is a test. It has multiple sentences. We want to detect plagiarism."),
        ("Slightly Modified Document",
         "This is a test. It has multiple sentences. We want to detect plagiarism.",
         "This is an exam. It contains several phrases. We aim to identify copying."),
        ("Completely Different Documents",
         "This is about cats. Cats are furry animals. They make good pets.",
         "Python is a programming language. It is widely used in data science."),
        ("Partial Overlap",
         "This is a test. It has multiple sentences. We want to detect plagiarism.",
         "This is different. We want to detect plagiarism. This is unique.")
    ]

    for test_name, doc1, doc2 in test_cases:
        print(f"\nTest Case: {test_name}")
        print(f"Document 1: {doc1}")
        print(f"Document 2: {doc2}")
        plagiarism_cases, overall_percentage = detect_plagiarism(doc1, doc2)

        print(f"Detected {len(plagiarism_cases)} plagiarism cases:")
        for i, j, similarity in plagiarism_cases:
            print(f"  Sentence {i + 1} in doc1 matches sentence {j + 1} in doc2 with {similarity:.2f}% similarity")

        verdict = plagiarism_verdict(overall_percentage)
        print(f"Overall similarity: {overall_percentage:.2f}%")
        print(f"Verdict: {verdict}")

if __name__ == "__main__":
    run_test_cases()
