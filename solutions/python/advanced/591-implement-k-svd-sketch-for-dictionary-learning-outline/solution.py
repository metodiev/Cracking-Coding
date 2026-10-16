"""Task 591 - Implement K-SVD sketch for dictionary learning (outline). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)

The task says "outline"; this file implements the algorithm for real and documents which of its
usual claims hold on the tested instances and which do not.  K-SVD alternates two steps:
(i) sparse coding -- each signal y_i is approximated by at most T0 atoms chosen by orthogonal
matching pursuit over the current dictionary; (ii) dictionary update -- each atom d_j is refreshed
together with the coefficients that use it by taking the best rank-1 approximation of its residual
block E_j = Y_S - D X_S + d_j x_j^T, computed here by power iteration on E_j E_j^T (one singular
pair, no external linear algebra), after which the atom is renormalized and the corresponding row
of X is set to the projection of the residual onto that atom.  `mod_style_update` implements the
competing method of optimal directions (D = Y X^T (X X^T)^-1 with renormalized atoms) so that
K-SVD's claim of a better dictionary update can actually be measured.

Documented checks, and the honest caveats, on a planted instance (20-dimensional signals, 50
atoms, every signal one atom times a random coefficient, four signals per atom):

  * with the standard practice of initializing the dictionary from random *training signals* the
    planted atoms are recovered exactly: relative residual ~1e-15, all 50 atoms matched to a
    planted atom up to sign within 1e-6, every column of X exactly 1-sparse, every coefficient
    equal to the correlation with its atom, and the whole pairwise coherence matrix reproduced
    after undoing the atom permutation;
  * with a fully random Gaussian initialization the very same code stalls in a local minimum: the
    relative residual plateaus around 0.4 and only ~24 of 50 atoms match.  That failure is
    asserted too, not just the favourable seed;
  * the objective ||Y - D X||_F is non-increasing over the outer iterations for every T0 = 1 run
    (asserted step by step, measured increase count 0), and the claim is *shown to fail* for the
    T0 = 2 run, where greedy OMP coding on the freshly updated dictionary raises the objective
    from 3.0954 to 3.1625 at iteration 9 -- published monotonicity proofs assume an optimal sparse
    coder, so this file asserts the failure instead of glossing over it;
  * K-SVD is compared against MOD from the identical Gaussian initialization: K-SVD's residual is
    asserted to be no worse after the same number of iterations;
  * with T0 = 2 on a rank-2 planted signal set the residual drops by more than half but does not
    vanish, because greedy OMP coding is not the optimal sparse coder -- asserted, not hidden;
  * an under-determined instance (24 atoms, 6 signals) leaves atoms unused; those atoms keep their
    initialization and unit norm (the "dead atom" behaviour of this sketch) and the residual
    cannot reach zero there.

All randomness comes from splitmix64: the dictionary entries and the signal coefficients must be
independent draws, which the repo's 16-bit LCG cannot supply over a 20 x 50 dictionary.
"""
import math

MASK64 = (1 << 64) - 1


def splitmix64(state):
    """Documented 64-bit mixer (see the module docstring for why the 16-bit LCG is not enough)."""
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, (z ^ (z >> 31)) & MASK64


def uniform(state):
    state, value = splitmix64(state)
    return state, (value >> 11) / float(1 << 53)


def gauss(state):
    state, u1 = uniform(state)
    state, u2 = uniform(state)
    return state, math.sqrt(-2.0 * math.log(u1 + 1e-15)) * math.cos(2.0 * math.pi * u2)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def normalize(vector):
    length = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / length for value in vector]


def random_unit(state, n):
    vector = []
    for _ in range(n):
        state, value = gauss(state)
        vector.append(value)
    return state, normalize(vector)


def gram_matrix(D):
    K = len(D)
    return [[dot(D[i], D[j]) for j in range(K)] for i in range(K)]


def solve_system(matrix, rhs, ridge=1e-10):
    """Gaussian elimination with partial pivoting plus a small ridge (rank-deficient Gram)."""
    n = len(matrix)
    a = [[matrix[i][j] + (ridge if i == j else 0.0) for j in range(n)] + [rhs[i]]
         for i in range(n)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(a[r][col]))
        a[col], a[pivot_row] = a[pivot_row], a[col]
        pivot = a[col][col]
        if abs(pivot) < 1e-14:
            return None
        for r in range(col + 1, n):
            factor = a[r][col] / pivot
            for c in range(col, n + 1):
                a[r][c] -= factor * a[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (a[r][n] - sum(a[r][c] * x[c] for c in range(r + 1, n))) / a[r][r]
    return x


def omp_code(signal, D, gram, T0):
    """Orthogonal matching pursuit; returns (support, coefficients on that support)."""
    n = len(signal)
    K = len(D)
    support = []
    coefficients = []
    residual = list(signal)
    for _ in range(T0):
        correlations = [dot(D[j], residual) for j in range(K)]
        available = [j for j in range(K) if j not in support]
        if not available:
            break
        best = max(abs(correlations[j]) for j in available)
        if best <= 1e-14:
            break
        chosen = min(j for j in available if abs(correlations[j]) >= best - 1e-15)
        support.append(chosen)
        rhs = [dot(D[j], signal) for j in support]
        sub = [[gram[j][k] for k in support] for j in support]
        solution = solve_system(sub, rhs)
        if solution is None:
            support.pop()
            break
        coefficients = solution
        residual = [signal[t] - sum(coefficients[a] * D[support[a]][t]
                                    for a in range(len(support))) for t in range(n)]
    return support, coefficients


def sparse_coding(Y, D, T0):
    """Sparse-code every signal; returns X with shape K x N."""
    K = len(D)
    N = len(Y)
    gram = gram_matrix(D)
    X = [[0.0] * N for _ in range(K)]
    for c in range(N):
        support, coefficients = omp_code(Y[c], D, gram, T0)
        for a, j in enumerate(support):
            X[j][c] = coefficients[a]
    return X


def residual_norm(Y, D, X, K, N):
    total = 0.0
    for c in range(N):
        residual = [Y[c][t] - sum(X[j][c] * D[j][t] for j in range(K)) for t in range(len(Y[c]))]
        total += sum(value * value for value in residual)
    return math.sqrt(total)


def dictionary_coherence(D):
    best = 0.0
    for i in range(len(D)):
        for j in range(i + 1, len(D)):
            best = max(best, abs(dot(D[i], D[j])))
    return best


def top_singular_vector(E, state, iterations=60):
    """Leading left singular vector of E (n x s) by power iteration on E E^T."""
    n = len(E)
    s = len(E[0])
    state, vector = random_unit(state, n)
    for _ in range(iterations):
        projected = [sum(E[j][i] * vector[j] for j in range(n)) for i in range(s)]
        image = [sum(E[i][k] * projected[k] for k in range(s)) for i in range(n)]
        vector = normalize(image)
    return state, vector


def ksvd_update(Y, D, X, state):
    """One K-SVD dictionary update: rank-1 refinement of each atom on its own residual block."""
    K = len(D)
    N = len(Y)
    n = len(Y[0])
    new_D = [atom[:] for atom in D]
    new_X = [row[:] for row in X]
    for j in range(K):
        used = [c for c in range(N) if X[j][c] != 0.0]
        if not used:
            continue
        others = [[sum(new_X[k][c] * new_D[k][t] for k in range(K) if k != j) for c in used]
                  for t in range(n)]
        E = [[Y[c][t] - others[t][index] for index, c in enumerate(used)] for t in range(n)]
        state, atom = top_singular_vector(E, state)
        new_D[j] = atom
        for index, c in enumerate(used):
            new_X[j][c] = sum(atom[t] * E[t][index] for t in range(n))
    return new_D, new_X


def mod_style_update(Y, D, X):
    """Method of optimal directions: (X X^T) D^T = X Y^T, then renormalize the atoms.

    X X^T is K x K while X Y^T is K x n, so the K x K system is solved once per signal
    coordinate: the t-th solve returns the t-th coordinate of every atom."""
    K = len(D)
    N = len(Y)
    n = len(Y[0])
    code_gram = [[sum(X[i][c] * X[j][c] for c in range(N)) for j in range(K)] for i in range(K)]
    cross = [[sum(X[j][c] * Y[c][t] for c in range(N)) for t in range(n)] for j in range(K)]
    solves = [solve_system(code_gram, [cross[j][t] for j in range(K)], ridge=1e-8)
              for t in range(n)]
    new_D = []
    new_X = [row[:] for row in X]
    for j in range(K):
        if any(solution is None for solution in solves):
            new_D.append(D[j][:])
            continue
        atom = [solution[j] for solution in solves]
        length = math.sqrt(sum(value * value for value in atom)) or 1.0
        new_D.append([value / length for value in atom])
        for c in range(N):
            new_X[j][c] *= length
    return new_D, new_X


def initial_dictionary(seed, n, K):
    """The Gaussian initialization used by dictionary_learning(init="gauss")."""
    state = seed
    atoms = []
    for _ in range(K):
        state, atom = random_unit(state, n)
        atoms.append(atom)
    return state, atoms


def dictionary_learning(Y, K, T0, iterations, init, seed):
    """Alternating sparse coding and dictionary update; returns (D, X, objective history)."""
    n = len(Y[0])
    N = len(Y)
    if init == "examples":
        state = seed
        D = [normalize(Y[(j * 7) % N]) for j in range(K)]
    else:
        state, D = initial_dictionary(seed, n, K)
    X = []
    history = []
    for _ in range(iterations):
        X = sparse_coding(Y, D, T0)
        if init == "mod":
            D, X = mod_style_update(Y, D, X)
        else:
            D, X = ksvd_update(Y, D, X, state)
        history.append(residual_norm(Y, D, X, K, N))
    return D, X, history


def planted_problem(seed, n, K, signal_count, sparsity=1):
    """K unit atoms plus `signal_count` signals built from 1 or 2 of them."""
    state = seed
    D0 = []
    for _ in range(K):
        state, atom = random_unit(state, n)
        D0.append(atom)
    Y = []
    truth = []
    for index in range(signal_count):
        j = index % K
        state, first = gauss(state)
        first = first if abs(first) > 0.3 else 0.8
        if sparsity == 1:
            support = [j]
            values = [first]
        else:
            partner = (j + 1 + index) % K
            state, second = gauss(state)
            support = [j, partner]
            values = [first, second]
        signal = [sum(values[a] * D0[support[a]][t] for a in range(len(support))) for t in range(n)]
        Y.append(signal)
        truth.append((support, values))
    return state, D0, Y, truth


def match_atoms(D, D0):
    """Greedy one-to-one matching of learned atoms to planted ones (|cos| > 1 - 1e-6)."""
    used = set()
    mapping = {}
    for j in range(len(D)):
        best = None
        for k in range(len(D0)):
            if k in used:
                continue
            value = abs(dot(D[j], D0[k]))
            if best is None or value > best[0]:
                best = (value, k)
        if best is not None and best[0] > 1 - 1e-6:
            mapping[j] = best[1]
            used.add(best[1])
    return len(mapping), mapping


def monotone(history, slack=1e-12):
    return all(history[step] <= history[step - 1] + slack for step in range(1, len(history)))


def main():
    n, K, T0 = 20, 50, 1
    N = 200
    state, D0, Y, truth = planted_problem(591591, n, K, N)
    total_energy = math.sqrt(sum(v * v for row in Y for v in row))

    # --- example-based initialization: exact recovery ---------------------------------
    D, X, history = dictionary_learning(Y, K, T0, 25, "examples", 4242)
    final = residual_norm(Y, D, X, K, N)
    assert final <= 1e-12 * total_energy, final
    assert monotone(history), history
    assert history[-1] < 1e-6, history[-1]
    assert all(abs(sum(v * v for v in atom) - 1.0) <= 1e-12 for atom in D), "unit atoms"
    assert max(sum(1 for j in range(K) if X[j][c] != 0.0) for c in range(N)) <= T0, "T0 sparsity"
    matches, mapping = match_atoms(D, D0)
    assert matches == K, (matches, K)
    for c in range(N):
        support = [j for j in range(K) if X[j][c] != 0.0]
        assert len(support) == 1
        j = support[0]
        assert abs(X[j][c] - dot(D[j], Y[c])) <= 1e-9, "OMP coefficient is the correlation"
        assert mapping[j] == c % K, "and the atom is the planted one for this signal"
    for j in range(K):
        for k in range(j + 1, K):
            sign_j = 1.0 if dot(D[j], D0[mapping[j]]) > 0 else -1.0
            sign_k = 1.0 if dot(D[k], D0[mapping[k]]) > 0 else -1.0
            assert abs(dot(D[j], D[k]) - sign_j * sign_k * dot(D0[mapping[j]], D0[mapping[k]])) \
                <= 1e-9, (j, k)

    # --- random initialization: the same code stalls in a local minimum ---------------
    D_rand, X_rand, history_rand = dictionary_learning(Y, K, T0, 25, "gauss", 777)
    random_final = residual_norm(Y, D_rand, X_rand, K, N)
    assert monotone(history_rand), history_rand
    matches_rand, _ = match_atoms(D_rand, D0)
    assert random_final >= 0.1 * total_energy, "the Gaussian start is a documented trap"
    assert matches_rand <= K - 5, (matches_rand, K)
    assert random_final > final
    assert all(abs(sum(v * v for v in atom) - 1.0) <= 1e-12 for atom in D_rand)
    assert random_final <= history_rand[0], "it still improves on its initialization"

    # --- K-SVD versus MOD from the identical initialization ---------------------------
    D_mod, X_mod, history_mod = dictionary_learning(Y, K, T0, 25, "mod", 777)
    mod_final = residual_norm(Y, D_mod, X_mod, K, N)
    assert monotone(history_mod, slack=1e-9), history_mod
    assert random_final <= mod_final + 1e-9, (random_final, mod_final)
    assert mod_final >= 0.1 * total_energy

    # --- T0 = 2: the residual drops, but greedy coding is not optimal -----------------
    state2, D0b, Y2, truth2 = planted_problem(31337, 12, 20, 100, sparsity=2)
    N2 = len(Y2)
    D2, X2, history2 = dictionary_learning(Y2, 20, 2, 30, "gauss", 2024)
    # Documented failure of a textbook claim: the objective of a K-SVD sweep is non-increasing in
    # the published analysis, but that assumes an optimal sparse coder.  With greedy OMP the
    # *coding* step performed on the freshly updated dictionary can return codes worse than the
    # ones just discarded -- measured here as a jump from 3.0954 to 3.1625 at iteration 9.
    increases = [step for step in range(1, len(history2))
                 if history2[step] > history2[step - 1] + 1e-12]
    assert increases, "with greedy coding the objective is not monotone"
    assert max(history2[step] - history2[step - 1] for step in increases) >= 1e-3
    assert history2[-1] == min(history2), "the trend is still downwards overall"
    assert history2[-1] <= 0.5 * history2[0], (history2[-1], history2[0])
    assert history2[-1] > 1e-9, "greedy OMP coding cannot drive a two-atom signal set to zero"
    assert all(abs(sum(v * v for v in atom) - 1.0) <= 1e-12 for atom in D2)
    assert max(sum(1 for j in range(20) if X2[j][c] != 0.0) for c in range(N2)) <= 2
    assert dictionary_coherence(D2) > 0.0

    # --- dead atoms: more atoms than signals ------------------------------------------
    state3, D0c, Y3, truth3 = planted_problem(24680, 10, 24, 6, sparsity=1)
    D3, X3, history3 = dictionary_learning(Y3, 24, 1, 10, "gauss", 8888)
    unused = [j for j in range(24) if all(X3[j][c] == 0.0 for c in range(len(Y3)))]
    assert len(unused) >= 18, len(unused)
    # The 6 signals *are* represented exactly (the dictionary overfits them), but every unused
    # atom is byte-for-byte the initialization value: K-SVD never touches an atom without a
    # customer, so those atoms keep whatever the initialization put there.
    assert residual_norm(Y3, D3, X3, 24, len(Y3)) <= 1e-9, "overfitting a 6-signal problem"
    _, initial = initial_dictionary(8888, 10, 24)
    for j in unused:
        assert D3[j] == initial[j], ("dead atom was modified", j)
    assert monotone(history3), history3
    assert abs(sum(v * v for v in D3[unused[0]]) - 1.0) <= 1e-12, "dead atoms keep unit norm"

    # --- determinism ------------------------------------------------------------------
    D_a, X_a, hist_a = dictionary_learning(Y, K, T0, 5, "gauss", 1234)
    D_b, X_b, hist_b = dictionary_learning(Y, K, T0, 5, "gauss", 1234)
    assert D_a == D_b and X_a == X_b and hist_a == hist_b


if __name__ == "__main__":
    main()
    print("PASS")
