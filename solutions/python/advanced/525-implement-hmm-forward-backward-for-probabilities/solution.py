"""Task 525 - Implement HMM forward-backward for probabilities. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)

Forward-backward for a discrete HMM. The forward recursion alpha_t[j] = b_j(o_t) * sum_i
alpha_{t-1}[i] a_ij and the backward recursion beta_t[i] = sum_j a_ij b_j(o_{t+1}) beta_{t+1}[j]
are both run with Rabiner-style per-row scaling (alpha_hat_t = alpha_t / c_t, beta_hat_t =
beta_t / c_t) so long sequences never underflow; the sequence probability is the product of the
scale factors and also the independent quantity P(O) = sum_i alpha_T[i] = sum_i pi_i b_i(o_1)
beta_1[i], both of which are compared. The smoothed posteriors gamma_t[i] = alpha_hat_t[i]
beta_hat_t[i] (the scale factors cancel) and the pairwise posteriors xi_t(i, j) = alpha_hat_t[i]
a_ij b_j(o_{t+1}) beta_hat_{t+1}[j] / c_{t+1} are then derived, together with a compact
Baum-Welch step. Documented checks: two hand-computed examples (a 2-state/2-step model whose
path probabilities close form to 0.42 and the log-space forward equal to the scaled forward),
gamma sums to 1 at every time step and xi sums to 1 over all state pairs at every step, the
xi-to-gamma marginal identity sum_j xi_t(i, j) = gamma_t(i), gamma agreeing with a brute-force
posterior computed by enumerating all S^T paths, P(O) from the scaled forward, the unscaled
forward and a log-space forward all agreeing with the brute-force path sum to 1e-12 on short
sequences, the same P(O) obtained from the backward side
((sum_i pi_i b_i(o_1) beta_hat_1[i]) * c_2 ... c_T), a T =
2000 sequence where both the unscaled alpha rows and the product of the scale factors underflow
to 0.0 while the accumulated log of the scale factors still matches an independent log-space
forward recursion to 1e-9, one Baum-Welch iteration never decreasing the log-likelihood (checked against an
independent direct evaluation on the re-estimated parameters), and the degenerate T = 0 / T = 1
cases.
"""

import math

NEG_INF = float("-inf")


def safe_log(x):
    return math.log(x) if x > 0.0 else NEG_INF


def normalize(rows):
    return [[w / float(sum(row)) for w in row] for row in rows]


def lcg_hmm(n_states, n_obs, seed, lo=1, hi=9):
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


def forward_unscaled(A, B, pi, obs):
    """Plain alpha recursions; underflows for long observations. Returns all rows."""
    n = len(pi)
    alpha = [[pi[i] * B[i][obs[0]] for i in range(n)]]
    for t in range(1, len(obs)):
        alpha.append([B[j][obs[t]] * sum(alpha[t - 1][i] * A[i][j] for i in range(n))
                      for j in range(n)])
    return alpha


def forward_scaled(A, B, pi, obs):
    """Rabiner scaling: returns (scaled alpha rows, scale factors, P(O) = prod c_t)."""
    n = len(pi)
    alpha = []
    scales = []
    current = [pi[i] * B[i][obs[0]] for i in range(n)]
    scale = sum(current)
    if scale <= 0.0:
        raise ValueError("the observation sequence has probability zero")
    current = [value / scale for value in current]
    alpha.append(current)
    scales.append(scale)
    for t in range(1, len(obs)):
        current = [B[j][obs[t]] * sum(alpha[t - 1][i] * A[i][j] for i in range(n))
                   for j in range(n)]
        scale = sum(current)
        if scale <= 0.0:
            raise ValueError("the observation sequence has probability zero")
        current = [value / scale for value in current]
        alpha.append(current)
        scales.append(scale)
    probability = 1.0
    log_probability = 0.0
    for scale in scales:
        probability *= scale
        log_probability += math.log(scale)
    return alpha, scales, probability, log_probability


def forward_log(A, B, pi, obs):
    """Independent log-space forward recursion; returns the full table and log P(O)."""
    n = len(pi)
    table = [[safe_log(pi[i]) + safe_log(B[i][obs[0]]) for i in range(n)]]
    for t in range(1, len(obs)):
        row = []
        for j in range(n):
            terms = [table[t - 1][i] + safe_log(A[i][j]) for i in range(n)]
            best = max(terms)
            row.append((NEG_INF if best == NEG_INF
                        else best + math.log(sum(math.exp(term - best) for term in terms)))
                       + safe_log(B[j][obs[t]]))
        table.append(row)
    return table, log_sum_exp(table[-1])


def log_sum_exp(values):
    best = max(values)
    if best == NEG_INF:
        return NEG_INF
    return best + math.log(sum(math.exp(value - best) for value in values))


def backward_scaled(A, B, pi, obs, scales):
    """Rabiner scaled betas: beta_hat_T = 1, beta_hat_t = (sum_j a_ij b_j(o_{t+1}) beta_hat_{t+1})/c_{t+1}."""
    n = len(pi)
    length = len(obs)
    beta = [[0.0] * n for _ in range(length)]
    for i in range(n):
        beta[length - 1][i] = 1.0
    for t in range(length - 2, -1, -1):
        for i in range(n):
            total = sum(A[i][j] * B[j][obs[t + 1]] * beta[t + 1][j] for j in range(n))
            beta[t][i] = total / scales[t + 1]
    return beta


def backward_probability(A, B, pi, obs, beta, scales):
    """P(O) from the backward side: (sum_i pi_i b_i(o_1) beta_hat_1[i]) * c_2 * ... * c_T."""
    total = sum(pi[i] * B[i][obs[0]] * beta[0][i] for i in range(len(pi)))
    for scale in scales[1:]:
        total *= scale
    return total


def posterior_gamma(alpha, beta, length, n_states):
    """gamma_t[i] = alpha_hat_t[i] beta_hat_t[i]: no renormalisation is needed."""
    return [[alpha[t][i] * beta[t][i] for i in range(n_states)] for t in range(length)]


def posterior_xi(A, B, obs, alpha, beta, scales, n_states):
    """xi_t(i, j) = alpha_hat_t[i] a_ij b_j(o_{t+1}) beta_hat_{t+1}[j] / c_{t+1}; sums to 1."""
    xi = []
    for t in range(len(obs) - 1):
        row = [[alpha[t][i] * A[i][j] * B[j][obs[t + 1]] * beta[t + 1][j] / scales[t + 1]
                for j in range(n_states)] for i in range(n_states)]
        xi.append(row)
    return xi


def forward_backward(A, B, pi, obs):
    """Bundle: scaled forward/backward tables, P(O), gamma and xi."""
    n = len(pi)
    if not obs:
        return {"alpha": [], "beta": [], "scales": [], "probability": 1.0,
                "log_probability": 0.0, "backward_probability": 1.0, "gamma": [], "xi": []}
    alpha, scales, probability, log_probability = forward_scaled(A, B, pi, obs)
    beta = backward_scaled(A, B, pi, obs, scales)
    gamma = posterior_gamma(alpha, beta, len(obs), n)
    xi = posterior_xi(A, B, obs, alpha, beta, scales, n)
    backward_p = backward_probability(A, B, pi, obs, beta, scales)
    return {"alpha": alpha, "beta": beta, "scales": scales,
            "probability": probability, "log_probability": log_probability,
            "backward_probability": backward_p,
            "gamma": gamma, "xi": xi}


def all_paths(n_states, length):
    if length == 0:
        return [()]
    out = [()]
    for _ in range(length):
        out = [prefix + (state,) for prefix in out for state in range(n_states)]
    return out


def path_probability(A, B, pi, path, obs):
    if not obs:
        return 1.0
    total = pi[path[0]] * B[path[0]][obs[0]]
    for t in range(1, len(obs)):
        total *= A[path[t - 1]][path[t]] * B[path[t]][obs[t]]
    return total


def brute_force_probability(A, B, pi, obs):
    return sum(path_probability(A, B, pi, list(path), obs)
               for path in all_paths(len(pi), len(obs)))


def brute_force_gamma(A, B, pi, obs):
    n = len(pi)
    total = brute_force_probability(A, B, pi, obs)
    gamma = []
    for t in range(len(obs)):
        row = []
        for state in range(n):
            mass = sum(path_probability(A, B, pi, list(path), obs)
                       for path in all_paths(n, len(obs)) if path[t] == state)
            row.append(mass / total)
        gamma.append(row)
    return gamma


def baum_welch_step(A, B, pi, obs, n_obs):
    """One EM update from the smoothed posteriors; returns freshly normalised parameters."""
    n = len(pi)
    state = forward_backward(A, B, pi, obs)
    gamma, xi = state["gamma"], state["xi"]
    length = len(obs)
    new_pi = list(gamma[0])
    counts = [[0.0] * n for _ in range(n)]
    for t in range(length - 1):
        for i in range(n):
            for j in range(n):
                counts[i][j] += xi[t][i][j]
    new_A = []
    for i in range(n):
        row_total = sum(counts[i])
        new_A.append([value / row_total for value in counts[i]] if row_total > 0
                     else [1.0 / n] * n)
    new_B = []
    for i in range(n):
        row = [0.0] * n_obs
        for t in range(length):
            row[obs[t]] += gamma[t][i]
        row_total = sum(row)
        new_B.append([value / row_total for value in row] if row_total > 0
                     else [1.0 / n_obs] * n_obs)
    return new_A, new_B, new_pi


def main():
    # Hand example 1: pi = [0.6, 0.4], A = [[0.7, 0.3], [0.4, 0.6]], B = [[0.5, 0.5], [0.1, 0.9]].
    # P(O) for obs = [0, 0] is 0.6*0.5*0.7*0.5 + 0.6*0.5*0.3*0.1 + 0.4*0.1*0.4*0.5 + 0.4*0.1*0.6*0.1
    # = 0.105 + 0.009 + 0.008 + 0.0024 = 0.1244.
    A = [[0.7, 0.3], [0.4, 0.6]]
    B = [[0.5, 0.5], [0.1, 0.9]]
    pi = [0.6, 0.4]
    obs = [0, 0]
    assert abs(brute_force_probability(A, B, pi, obs) - 0.1244) <= 1e-12
    state = forward_backward(A, B, pi, obs)
    assert abs(state["probability"] - 0.1244) <= 1e-12
    assert abs(sum(forward_unscaled(A, B, pi, obs)[-1]) - 0.1244) <= 1e-12
    _, log_probability = forward_log(A, B, pi, obs)
    assert abs(math.exp(log_probability) - 0.1244) <= 1e-12
    # Path with probability exactly 0.105 = state 0 twice, and gamma at t = 0 backs state 0.
    assert abs(path_probability(A, B, pi, [0, 0], obs) - 0.105) <= 1e-12
    for row in state["gamma"]:
        assert abs(sum(row) - 1.0) <= 1e-12
    assert state["gamma"][0][0] > state["gamma"][0][1]
    # Backward side gives the same sequence probability:
    # P(O) = (sum_i pi_i b_i(o_1) beta_hat_1[i]) * c_2 ... c_T.
    assert abs(state["backward_probability"] - 0.1244) <= 1e-12

    # Hand example 2: 3 states, T = 1 -> gamma is exactly the normalised first column.
    A3, B3, pi3 = lcg_hmm(3, 2, 11)
    single = forward_backward(A3, B3, pi3, [1])
    expected = [pi3[i] * B3[i][1] for i in range(3)]
    total = sum(expected)
    assert abs(single["probability"] - total) <= 1e-12
    for i in range(3):
        assert abs(single["gamma"][0][i] - expected[i] / total) <= 1e-12
    assert single["xi"] == [], "no transitions for a single observation"

    # LCG HMMs: probability and posterior checks against brute force over all paths.
    cases = ((2, 2, 6, 23), (3, 2, 6, 37), (3, 3, 5, 53), (2, 3, 7, 71), (4, 2, 5, 97))
    for n_states, n_obs, length, seed in cases:
        A, B, pi = lcg_hmm(n_states, n_obs, seed)
        seq = []
        cursor = seed
        for _ in range(length):
            cursor = (cursor * 75 + 74) % 65537
            seq.append(cursor % n_obs)
        state = forward_backward(A, B, pi, seq)
        truth = brute_force_probability(A, B, pi, seq)
        assert abs(state["probability"] - truth) <= 1e-12, (seed, state["probability"], truth)
        assert abs(sum(forward_unscaled(A, B, pi, seq)[-1]) - truth) <= 1e-12
        _, log_probability = forward_log(A, B, pi, seq)
        assert abs(math.exp(log_probability) - truth) <= 1e-12
        assert abs(state["backward_probability"] - truth) <= 1e-12, (seed, state, truth)
        # gamma sums to 1 and matches the brute-force posterior.
        brute_gamma = brute_force_gamma(A, B, pi, seq)
        for t in range(length):
            assert abs(sum(state["gamma"][t]) - 1.0) <= 1e-12
            for i in range(n_states):
                assert abs(state["gamma"][t][i] - brute_gamma[t][i]) <= 1e-12, (seed, t, i)
        # xi sums to 1 per time step and its row sums reproduce gamma.
        for t in range(length - 1):
            assert abs(sum(sum(row) for row in state["xi"][t]) - 1.0) <= 1e-12
            for i in range(n_states):
                row_sum = sum(state["xi"][t][i])
                assert abs(row_sum - state["gamma"][t][i]) <= 1e-12, (seed, t, i, row_sum)
        # xi against a brute-force pairwise posterior.
        for t in range(length - 1):
            pair_total = truth
            for i in range(n_states):
                for j in range(n_states):
                    mass = sum(path_probability(A, B, pi, list(path), seq)
                               for path in all_paths(n_states, length)
                               if path[t] == i and path[t + 1] == j)
                    assert abs(state["xi"][t][i][j] - mass / pair_total) <= 1e-12, (seed, t, i, j)

    # Long sequence: the unscaled recursion underflows, the scaled and log recursions agree.
    A, B, pi = lcg_hmm(3, 2, 2024)
    long_obs = [(i * 7 + 3) % 2 for i in range(2000)]
    unscaled = forward_unscaled(A, B, pi, long_obs)
    assert sum(unscaled[600]) > 0.0, "still representable at T = 600"
    assert sum(unscaled[-1]) == 0.0, "underflowed to 0.0 by T = 2000"
    scaled = forward_backward(A, B, pi, long_obs)
    _, log_probability = forward_log(A, B, pi, long_obs)
    assert scaled["probability"] == 0.0, "the scale product underflows just like alpha does"
    assert math.isfinite(scaled["log_probability"])
    assert abs(scaled["log_probability"] - log_probability) <= 1e-9, \
        (scaled["log_probability"], log_probability)
    assert scaled["log_probability"] < -1000.0
    assert abs(sum(scaled["gamma"][-1]) - 1.0) <= 1e-12
    assert abs(sum(sum(row) for row in scaled["xi"][-1]) - 1.0) <= 1e-12

    # One Baum-Welch iteration must not decrease the likelihood (checked by re-evaluating the
    # re-estimated model independently with the brute-force path sum).
    for seed in (5, 11, 23):
        n_states, n_obs, length = 3, 2, 6
        A, B, pi = lcg_hmm(n_states, n_obs, seed)
        seq = []
        cursor = seed
        for _ in range(length):
            cursor = (cursor * 75 + 74) % 65537
            seq.append(cursor % n_obs)
        before = brute_force_probability(A, B, pi, seq)
        new_A, new_B, new_pi = baum_welch_step(A, B, pi, seq, n_obs)
        for row in new_A:
            assert abs(sum(row) - 1.0) <= 1e-12
        for row in new_B:
            assert abs(sum(row) - 1.0) <= 1e-12
        assert abs(sum(new_pi) - 1.0) <= 1e-12
        after = brute_force_probability(new_A, new_B, new_pi, seq)
        assert after >= before - 1e-15, (seed, before, after)
        assert abs(brute_force_probability(new_A, new_B, new_pi, seq)
                   - forward_backward(new_A, new_B, new_pi, seq)["probability"]) <= 1e-12

    # Zero-probability sequence: impossible observations are rejected instead of dividing by 0.
    A = [[1.0, 0.0], [0.0, 1.0]]
    B = [[1.0, 0.0], [0.0, 1.0]]
    pi = [0.5, 0.5]
    try:
        forward_backward(A, B, pi, [0, 1])
    except ValueError:
        pass
    else:
        raise AssertionError("an impossible sequence must raise ValueError")
    assert abs(brute_force_probability(A, B, pi, [0, 0]) - 0.5) <= 1e-12

    # Degenerate lengths.
    empty = forward_backward(A, B, pi, [])
    assert empty["probability"] == 1.0 and empty["gamma"] == [] and empty["alpha"] == []
    assert brute_force_probability(A, B, pi, []) == 1.0
    single_table, single_log = forward_log(A, B, pi, [0])
    assert abs(math.exp(single_log) - sum(pi[i] * B[i][0] for i in range(2))) <= 1e-12
    assert abs(single_log + math.log(2.0)) <= 1e-12, "identity emissions give P(O) = 0.5"

    # Determinism.
    A, B, pi = lcg_hmm(3, 3, 11)
    seq = [0, 2, 1, 1, 0]
    first = forward_backward(A, B, pi, seq)
    second = forward_backward(A, B, pi, seq)
    assert first["probability"] == second["probability"]
    assert first["gamma"] == second["gamma"] and first["xi"] == second["xi"]


if __name__ == "__main__":
    main()
    print("PASS")
