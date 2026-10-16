"""Task 524 - Implement Hidden Markov Model Viterbi algorithm. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)

Viterbi decoding of a discrete HMM (states, row-stochastic start vector pi, transition matrix A,
emission matrix B, observation sequence) in log space: delta_t[j] = max_i delta_{t-1}[i] + log A[i][j]
+ log B[j][o_t] with backpointers, then a backward walk from the argmax of the last column. Zero-probability entries are legal: log-space arithmetic uses -inf for them (the identity-emission
example below has zero emissions). Ties
are resolved deterministically by scanning predecessors in increasing state index with a strict
comparison, so the decoder returns the lexicographically smallest optimal path; the same
convention is used by the brute-force reference, which enumerates all S^T paths and keeps the
first maximum. Documented checks: two hand-computed examples (identity emissions give the unique
path [0, 1, 0] with probability 1/8; uniform emissions reduce the problem to transitions and give
the unique path [0, 0] with probability 0.105), exact agreement of the log-space score with the
brute-force maximum over all paths to 1e-12 for 3-state/T=8 and 2-state/T=10 LCG-parameterised
HMMs (the returned path must additionally equal the brute-force path whenever the optimum is
unique), an independent recomputation of the reported score from the returned path, agreement
with a separate linear-scale (product) DP on short sequences, Viterbi score <= total probability
P(O) = sum over all paths with a strict gap whenever several paths carry mass, the documented
classic example where the per-time posterior maximiser (computed here by brute force over paths)
differs from the jointly optimal path, an all-ties symmetric HMM returning the all-zero path, a
T = 2000 decoding that stays finite in log space while the product DP is finite only up to about
T = 600 and has underflowed to 0.0 by T = 1000, and the degenerate T = 0 / T = 1 cases.
"""

import math

NEG_INF = float("-inf")


def safe_log(x):
    """log with -inf for zero probability (zero entries are legal in A, B and pi)."""
    return math.log(x) if x > 0.0 else NEG_INF


def normalize(rows):
    return [[w / float(sum(row)) for w in row] for row in rows]


def lcg_hmm(n_states, n_obs, seed, lo=1, hi=9):
    """Deterministic HMM: integer weights in [lo, hi] normalised row-wise."""
    values = []
    for _ in range(n_states * n_states + n_states * n_obs + n_states):
        seed = (seed * 75 + 74) % 65537
        values.append(lo + seed % (hi - lo + 1))
    cursor = 0
    transitions = [values[cursor + i * n_states:cursor + (i + 1) * n_states]
                   for i in range(n_states)]
    cursor += n_states * n_states
    emissions = [values[cursor + i * n_obs:cursor + (i + 1) * n_obs] for i in range(n_states)]
    cursor += n_states * n_obs
    start = values[cursor:cursor + n_states]
    return normalize(transitions), normalize(emissions), normalize([start])[0]


def log_prob(A, B, pi, path, obs):
    """Independent scoring of one full state path: log pi[p0] + log b[p0][o0] + ... ."""
    if len(path) != len(obs):
        raise ValueError("path and observation sequence must have equal length")
    if not obs:
        return 0.0
    total = safe_log(pi[path[0]]) + safe_log(B[path[0]][obs[0]])
    for t in range(1, len(obs)):
        total += safe_log(A[path[t - 1]][path[t]]) + safe_log(B[path[t]][obs[t]])
    return total


def viterbi(A, B, pi, obs):
    """Log-space Viterbi; returns (lexicographically smallest optimal path, log score)."""
    n = len(pi)
    if not obs:
        return [], 0.0
    delta = [safe_log(pi[i]) + safe_log(B[i][obs[0]]) for i in range(n)]
    back = []
    for t in range(1, len(obs)):
        new_delta = [NEG_INF] * n
        pointers = [0] * n
        for j in range(n):
            best = NEG_INF
            best_state = 0
            for i in range(n):
                candidate = delta[i] + safe_log(A[i][j])
                if candidate > best:                     # strict: smallest index wins a tie
                    best = candidate
                    best_state = i
            new_delta[j] = best + safe_log(B[j][obs[t]])
            pointers[j] = best_state
        delta = new_delta
        back.append(pointers)
    last = 0
    for j in range(1, n):
        if delta[j] > delta[last]:
            last = j
    path = [last]
    for t in range(len(obs) - 2, -1, -1):
        path.append(back[t][path[-1]])
    path.reverse()
    return path, delta[last]


def viterbi_linear(A, B, pi, obs):
    """Second, independent implementation on plain probabilities (safe only for short obs)."""
    n = len(pi)
    if not obs:
        return [], 0.0
    delta = [pi[i] * B[i][obs[0]] for i in range(n)]
    back = []
    for t in range(1, len(obs)):
        new_delta = [0.0] * n
        pointers = [0] * n
        for j in range(n):
            best = -1.0
            best_state = 0
            for i in range(n):
                candidate = delta[i] * A[i][j]
                if candidate > best:
                    best = candidate
                    best_state = i
            new_delta[j] = best * B[j][obs[t]]
            pointers[j] = best_state
        delta = new_delta
        back.append(pointers)
    last = 0
    for j in range(1, n):
        if delta[j] > delta[last]:
            last = j
    path = [last]
    for t in range(len(obs) - 2, -1, -1):
        path.append(back[t][path[-1]])
    path.reverse()
    probability = delta[last]
    return path, (math.log(probability) if probability > 0.0 else NEG_INF)


def all_paths(n_states, length):
    """Lexicographic enumeration of all state sequences of the given length."""
    if length == 0:
        return [()]
    out = [()]
    for _ in range(length):
        out = [prefix + (state,) for prefix in out for state in range(n_states)]
    return out


def brute_force_best(A, B, pi, obs):
    """Maximum path score plus every lexicographically-ordered optimal path."""
    best = NEG_INF
    best_paths = []
    for path in all_paths(len(pi), len(obs)):
        score = log_prob(A, B, pi, path, obs)
        if score > best:
            best = score
            best_paths = [list(path)]
        elif score == best:
            best_paths.append(list(path))
    return best, best_paths


def total_probability(A, B, pi, obs):
    """P(O) = sum over all state paths, computed by brute force (independent of Viterbi)."""
    if not obs:
        return 1.0
    return sum(math.exp(log_prob(A, B, pi, list(path), obs))
               for path in all_paths(len(pi), len(obs)))


def posterior_paths(A, B, pi, obs):
    """Per-time posterior argmax path (brute force): the classic alternative to Viterbi."""
    n = len(pi)
    if not obs:
        return []
    weights = {}
    for path in all_paths(n, len(obs)):
        weights[path] = math.exp(log_prob(A, B, pi, list(path), obs))
    total = sum(weights.values())
    path = []
    for t in range(len(obs)):
        best_state = 0
        best_mass = -1.0
        for state in range(n):
            mass = sum(w for candidate, w in weights.items() if candidate[t] == state) / total
            if mass > best_mass:
                best_mass = mass
                best_state = state
        path.append(best_state)
    return path


def main():
    # Hand example 1: identity emissions pin the path; 2 states, A = [[.5,.5],[.5,.5]].
    A = [[0.5, 0.5], [0.5, 0.5]]
    B = [[1.0, 0.0], [0.0, 1.0]]
    pi = [0.5, 0.5]
    obs = [0, 1, 0]
    path, score = viterbi(A, B, pi, obs)
    assert path == [0, 1, 0], path
    assert abs(score - 3.0 * math.log(0.5)) <= 1e-12, score
    assert abs(math.exp(score) - 0.125) <= 1e-12
    best, optimal = brute_force_best(A, B, pi, obs)
    assert abs(score - best) <= 1e-12 and optimal == [[0, 1, 0]], "unique optimum"

    # Hand example 2: uniform emissions - only the transition weights matter.
    A = [[0.7, 0.3], [0.4, 0.6]]
    B = [[0.5, 0.5], [0.5, 0.5]]
    pi = [0.6, 0.4]
    path, score = viterbi(A, B, pi, [0, 0])
    assert path == [0, 0], path
    expected = math.log(0.6) + 2.0 * math.log(0.5) + math.log(0.7)
    assert abs(score - expected) <= 1e-12, (score, expected)
    assert abs(math.exp(score) - 0.105) <= 1e-12
    # The transition weights are what make state 1 second-best here: 0.4 * 0.4 * 0.25 = 0.04.
    assert abs(math.exp(log_prob(A, B, pi, [1, 0], [0, 0])) - 0.04) <= 1e-12

    # LCG HMMs: the log-space score equals the brute-force maximum and the path is optimal.
    cases = ((3, 2, 8, 11), (2, 3, 10, 23), (3, 3, 8, 37), (4, 2, 7, 53), (2, 2, 12, 71))
    for n_states, n_obs, length, seed in cases:
        A, B, pi = lcg_hmm(n_states, n_obs, seed)
        observations = []
        state = seed
        for _ in range(length):
            state = (state * 75 + 74) % 65537
            observations.append(state % n_obs)
        path, score = viterbi(A, B, pi, observations)
        assert len(path) == length and all(0 <= s < n_states for s in path)
        assert abs(log_prob(A, B, pi, path, observations) - score) <= 1e-9, \
            "reported score must match a direct evaluation of the returned path"
        best, optimal = brute_force_best(A, B, pi, observations)
        assert abs(score - best) <= 1e-12, (seed, score, best)
        assert path in optimal, (seed, path, optimal)
        if len(optimal) == 1:
            assert path == optimal[0], (seed, path, optimal)
        # Viterbi is never better than the full probability mass, and is strictly worse whenever
        # a second path carries positive probability.
        total = total_probability(A, B, pi, observations)
        assert math.exp(score) <= total + 1e-15, seed
        if len(optimal) > 1 or n_states > 1:
            assert math.exp(score) < total, seed
        # The linear-scale DP agrees on these short sequences.
        linear_path, linear_score = viterbi_linear(A, B, pi, observations)
        assert linear_path == path and abs(linear_score - score) <= 1e-9, seed

    # Documented tie-breaking: a fully symmetric HMM has every path optimal, and the decoder
    # returns the lexicographically smallest one, as does the brute-force reference.
    A = [[0.5, 0.5], [0.5, 0.5]]
    B = [[0.5, 0.5], [0.5, 0.5]]
    pi = [0.5, 0.5]
    path, score = viterbi(A, B, pi, [0, 1, 1, 0, 1])
    assert path == [0, 0, 0, 0, 0], path
    assert abs(score - 5.0 * math.log(0.25)) <= 1e-12
    best, optimal = brute_force_best(A, B, pi, [0, 1, 1, 0, 1])
    assert len(optimal) == 2 ** 5 and optimal[0] == [0, 0, 0, 0, 0]

    # Classic posterior-vs-Viterbi example: the per-time posterior maximiser is [0, 1, 0, 0]
    # while the jointly optimal path is [0, 1, 0, 1] (A = [[2/5, 3/5], [3/5, 2/5]],
    # B = [[4/11, 7/11], [2/5, 3/5]], pi = [3/5, 2/5], obs = [1, 1, 1, 1]).
    A = [[2.0 / 5.0, 3.0 / 5.0], [3.0 / 5.0, 2.0 / 5.0]]
    B = [[4.0 / 11.0, 7.0 / 11.0], [2.0 / 5.0, 3.0 / 5.0]]
    pi = [3.0 / 5.0, 2.0 / 5.0]
    obs = [1, 1, 1, 1]
    viterbi_path, viterbi_score = viterbi(A, B, pi, obs)
    posterior = posterior_paths(A, B, pi, obs)
    assert viterbi_path == [0, 1, 0, 1], viterbi_path
    assert posterior == [0, 1, 0, 0], posterior
    assert viterbi_path != posterior
    best, optimal = brute_force_best(A, B, pi, obs)
    assert abs(viterbi_score - best) <= 1e-12
    assert log_prob(A, B, pi, posterior, obs) < viterbi_score, \
        "the posterior path is less probable than the Viterbi path"

    # Long sequence: the linear DP underflows while the log-space decoder stays finite.
    A, B, pi = lcg_hmm(3, 2, 97)
    long_obs = [(i * 7 + 3) % 2 for i in range(2000)]
    log_path, log_score = viterbi(A, B, pi, long_obs)
    assert math.isfinite(log_score) and log_score < 0.0
    assert len(log_path) == 2000
    assert abs(log_prob(A, B, pi, log_path, long_obs) - log_score) <= 1e-6
    # Measured: the product DP is still finite at T = 600 (log score -697.7) and has underflowed
    # to 0.0 at T = 1000 (log score about -1163), well below the double-precision floor.
    assert math.isfinite(viterbi_linear(A, B, pi, long_obs[:600])[1])
    underflow_path, underflow_score = viterbi_linear(A, B, pi, long_obs[:1000])
    assert underflow_score == NEG_INF, "the product formulation has underflowed by T = 1000"
    assert log_prob(A, B, pi, underflow_path, long_obs[:1000]) < -700.0

    # Degenerate lengths.
    assert viterbi(A, B, pi, []) == ([], 0.0)
    assert brute_force_best(A, B, pi, []) == (0.0, [[]])
    path, score = viterbi(A, B, pi, [1])
    assert path == [max(range(3), key=lambda i: (pi[i] * B[i][1], -i))]
    assert abs(score - max(safe_log(pi[i]) + safe_log(B[i][1]) for i in range(3))) <= 1e-12

    # Determinism: repeated decoding is bit-identical.
    A, B, pi = lcg_hmm(3, 3, 11)
    obs = [0, 2, 1, 1, 0, 2, 2, 1]
    assert viterbi(A, B, pi, obs) == viterbi(A, B, pi, obs)

    # Malformed input: a path of the wrong length is rejected.
    try:
        log_prob(A, B, pi, [0, 1], obs)
    except ValueError:
        pass
    else:
        raise AssertionError("length mismatch must raise ValueError")


if __name__ == "__main__":
    main()
    print("PASS")
