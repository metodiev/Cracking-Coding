"""Task 730 - Implement randomized numerical linear algebra low-rank approx demo.  (Tier: Expert, CTCI Beyond — Numerical & Scientific Computing)

A rank-k approximation of an m x n matrix normally costs a full SVD.  Randomized numerical
linear algebra replaces the expensive part by a *sketch*: multiply by a random n x k Gaussian
matrix Omega, orthonormalise the result (Y = A Omega, Q = orth(Y)), project A onto that subspace
(B = Q^T A, only k x n) and take the SVD of the small B.  With k + p columns of oversampling the
range of Q captures the dominant subspace with high probability, so

    A ~= Q (Q^T A) = U_k Sigma_k V_k^T

matches the best possible rank-k approximation up to a small factor.

Everything here is hand-rolled so the demo is self-contained: the Gaussian sketch comes from a
fixed-seed Box-Muller transform, orthonormalisation is modified Gram-Schmidt, and the small SVD
is built from a cyclic Jacobi eigensolver applied to B B^T (the SVD of a small matrix is a
symmetric eigenproblem).  A second Jacobi pass on A^T A gives the *exact* SVD, so the
randomized approximation can be compared with the theoretical optimum instead of a heuristic.

Verified invariants: the Jacobi solver reproduces M V = V Lambda and an orthonormal V, the exact
SVD satisfies A = U Sigma V^T and its singular values match a diagonal test matrix, a matrix of
numerical rank r is recovered to rounding when k >= r while k = r - 1 fails visibly, the
randomized error respects the (1 + eps) * best-rank-k bound on decaying spectra, Q is orthonormal,
and the boundary cases (k = 0, k = n, the zero matrix) behave.

Simplified: one fixed sketch (Gaussian, no power iteration for slowly decaying spectra), no
blocked or pivoted QR, Jacobi rather than bidiagonalisation for the small SVD, and matrices small
enough to keep the exact SVD cheap - tens of rows rather than thousands.
"""
import math

MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def expect_value_error(call, message):
    try: call()
    except ValueError: return
    raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def uniform(state):
    state, r = splitmix64(state)
    return state, (r >> 11) / float(1 << 53)


def gaussian_matrix(rows, cols, seed):
    """Standard normal entries via Box-Muller on the deterministic stream."""
    state, out = seed & MASK64, []
    for _ in range(rows):
        row = []
        for _ in range(cols):
            state, u1 = uniform(state)
            state, u2 = uniform(state)
            radius = math.sqrt(-2.0 * math.log(max(u1, 1e-18)))
            row.append(radius * math.cos(2.0 * math.pi * u2))
        out.append(row)
    return out, state


def identity(size):
    return [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]


def transpose(matrix):
    return [[matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix[0]))]


def matmul(left, right):
    rows, inner, cols = len(left), len(right), len(right[0])
    return [[sum(left[i][t] * right[t][j] for t in range(inner)) for j in range(cols)]
            for i in range(rows)]


def frobenius(matrix):
    return math.sqrt(sum(value * value for row in matrix for value in row))


def jacobi_eigen(symmetric, sweeps=60, tolerance=1e-15):
    """Cyclic Jacobi: returns (eigenvalues, eigenvectors) with eigenvectors as columns."""
    n = len(symmetric)
    a = [row[:] for row in symmetric]
    v = identity(n)
    for _ in range(sweeps):
        off = math.sqrt(sum(a[p][q] ** 2 for p in range(n) for q in range(p + 1, n)))
        if off <= tolerance * max(1.0, frobenius(a)):
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a[p][q]) <= 1e-300:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                sign = 1.0 if theta >= 0 else -1.0
                t = sign / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    order = sorted(range(n), key=lambda i: -a[i][i])
    values = [a[i][i] for i in order]
    vectors = [[v[row][i] for i in order] for row in range(n)]
    return values, vectors


def exact_svd(matrix):
    """SVD from the eigendecomposition of A^T A; singular values sorted descending."""
    at = transpose(matrix)
    gram = matmul(at, matrix)
    values, vectors = jacobi_eigen(gram)
    sigma = [math.sqrt(max(value, 0.0)) for value in values]
    threshold = 1e-6 * sigma[0] if sigma else 0.0
    keep = [i for i, value in enumerate(sigma) if value > threshold]
    right = [[vectors[row][i] for i in keep] for row in range(len(vectors))]
    left = []
    for col, index in enumerate(keep):
        column = [sum(matrix[i][t] * right[t][col] for t in range(len(right)))
                  / sigma[index] for i in range(len(matrix))]
        left.append(column)
    left = [[left[col][i] for col in range(len(left))] for i in range(len(matrix))]
    return left, [sigma[i] for i in keep], right


def modified_gram_schmidt(matrix):
    """Orthonormal basis of the columns of `matrix`, dropping directions that collapse."""
    rows = len(matrix)
    cols = len(matrix[0])
    basis = []
    for col in range(cols):
        vector = [matrix[i][col] for i in range(rows)]
        for previous in basis:
            projection = sum(previous[i] * vector[i] for i in range(rows))
            vector = [vector[i] - projection * previous[i] for i in range(rows)]
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 1e-10 * max(1.0, math.sqrt(rows)):
            basis.append([value / norm for value in vector])
    return [[basis[col][i] for col in range(len(basis))] for i in range(rows)]


def randomized_range(matrix, rank, seed, oversample=5, power_iterations=0):
    """Q = orth((A A^T)^q A Omega): an orthonormal basis of a random oversampled range."""
    rows, cols = len(matrix), len(matrix[0])
    if rank < 0 or rank > min(rows, cols):
        raise ValueError("rank must fit the matrix")
    if rank == 0:
        return [[0.0] for _ in range(rows)]
    width = min(rank + oversample, min(rows, cols))
    sketch, _ = gaussian_matrix(cols, width, seed)
    y = matmul(matrix, sketch)
    for _ in range(power_iterations):
        y = matmul(matrix, matmul(transpose(matrix), y))
    return modified_gram_schmidt(y)


def randomized_low_rank(matrix, rank, seed, oversample=5, power_iterations=0):
    """Rank-`rank` approximation Q (Q^T A), using the small SVD of the projected matrix."""
    if rank < 0:
        raise ValueError("rank must be non-negative")
    if rank == 0:
        return [[0.0] * len(matrix[0]) for _ in range(len(matrix))]
    q = randomized_range(matrix, rank, seed, oversample, power_iterations)
    if not q or not q[0]:
        return [[0.0] * len(matrix[0]) for _ in range(len(matrix))]
    projected = matmul(transpose(q), matrix)          # width x n
    left_small, sigma, right = exact_svd(projected)
    keep = min(rank, len(sigma))
    left = matmul(q, [[left_small[i][t] for t in range(keep)] for i in range(len(left_small))])
    return [[sum(left[i][t] * sigma[t] * right[j][t] for t in range(keep))
             for j in range(len(matrix[0]))] for i in range(len(matrix))]


def best_rank_k_error(matrix, rank):
    """Oracle: the exact tail sigma_{rank+1}.. computed from the full SVD."""
    _, sigma, _ = exact_svd(matrix)
    return math.sqrt(sum(value * value for value in sigma[rank:]))


def approximation_error(matrix, approximation):
    difference = [[matrix[i][j] - approximation[i][j] for j in range(len(matrix[0]))]
                  for i in range(len(matrix))]
    return frobenius(difference)


def low_rank_matrix(rows, cols, rank, seed, decay=0.6):
    """A x B with A (rows x rank) and B (rank x cols) Gaussian; spectrum set by the scale."""
    left, state = gaussian_matrix(rows, rank, seed)
    right, state = gaussian_matrix(rank, cols, state)
    scaled = [[value * (decay ** i) for value in row] for i, row in enumerate(left)]
    return matmul(scaled, right)


def main():
    # 1. the eigensolver is the foundation: M V = V Lambda and V must be orthonormal, checked
    #    against a symmetric matrix assembled from a random product.
    base, seed = gaussian_matrix(6, 4, 0x7301)
    symmetric = matmul(base, transpose(base))
    values, vectors = jacobi_eigen(symmetric)
    check(values == sorted(values, reverse=True), "eigenvalues must come out sorted")
    residual_matrix = matmul(symmetric, vectors)
    error = 0.0
    for i in range(6):
        for j in range(6):
            target = vectors[i][j] * values[j]
            error = max(error, abs(residual_matrix[i][j] - target))
    check(error < 1e-9, "M V must equal V Lambda")
    gram = matmul(transpose(vectors), vectors)
    check(max(abs(gram[i][j] - (1.0 if i == j else 0.0)) for i in range(6) for j in range(6))
          < 1e-10, "the eigenvector matrix must be orthonormal")
    diagonal_values, diagonal_vectors = jacobi_eigen([[3.0, 0.0], [0.0, 1.0]])
    check(abs(diagonal_values[0] - 3.0) < 1e-12 and abs(diagonal_values[1] - 1.0) < 1e-12,
          "a diagonal matrix must return its diagonal as eigenvalues")

    # 2. the exact SVD: singular values of a diagonal matrix and the reconstruction identity.
    left, sigma, right = exact_svd([[3.0, 0.0], [0.0, 1.0]])
    check(abs(sigma[0] - 3.0) < 1e-10 and abs(sigma[1] - 1.0) < 1e-10,
          "the diagonal singular values must be recovered")
    reconstruction = [[sum(left[i][t] * sigma[t] * right[j][t] for t in range(len(sigma)))
                       for j in range(2)] for i in range(2)]
    check(max(abs(reconstruction[i][j] - [[3.0, 0.0], [0.0, 1.0]][i][j])
              for i in range(2) for j in range(2)) < 1e-10, "A must equal U Sigma V^T")
    dense, seed = gaussian_matrix(7, 5, seed)
    left, sigma, right = exact_svd(dense)
    rebuild = [[sum(left[i][t] * sigma[t] * right[j][t] for t in range(len(sigma)))
                for j in range(5)] for i in range(7)]
    check(approximation_error(dense, rebuild) < 1e-9, "the exact SVD must reconstruct the matrix")
    check(all(sigma[i] >= sigma[i + 1] for i in range(len(sigma) - 1)),
          "singular values must be sorted descending")

    # 3. a matrix of numerical rank r: the randomized approximation is exact for k >= r and
    #    visibly wrong one rank lower, which pins the range finder to the true subspace.
    low = low_rank_matrix(20, 12, 3, 0x7302, decay=1.0)
    check(best_rank_k_error(low, 3) < 1e-8, "the test matrix really has rank three")
    for rank in (3, 5, 8):
        approximation = randomized_low_rank(low, rank, 0x7303)
        check(approximation_error(low, approximation) < 1e-8,
              f"rank {rank} must recover a rank-three matrix exactly")
    coarse = randomized_low_rank(low, 2, 0x7303)
    check(approximation_error(low, coarse) > 1e-3,
          "rank two must visibly fail on a rank-three matrix")

    # 4. the randomized error must respect the (1 + eps) best-rank-k bound on a decaying
    #    spectrum, and must decrease monotonically as the rank grows.
    decaying = low_rank_matrix(24, 16, 10, 0x7304, decay=0.55)
    errors, bounds = [], []
    for rank in (1, 2, 4, 6, 8):
        errors.append(approximation_error(decaying, randomized_low_rank(decaying, rank, 0x7305)))
        bounds.append(best_rank_k_error(decaying, rank))
    check(all(errors[i] > errors[i + 1] for i in range(len(errors) - 1)),
          "the approximation error must fall as the rank grows")
    check(all(errors[i] <= 1.5 * bounds[i] + 1e-9 for i in range(len(errors))),
          "the randomized error must be within 50% of the best rank-k error")
    check(bounds[-1] < bounds[0] / 100.0, "the tail must shrink by two orders of magnitude")

    # 5. the range basis must be orthonormal, oversampling must be what makes the bound hold,
    #    and a power iteration must not hurt.
    q = randomized_range(decaying, 4, 0x7305)
    width = len(q[0])
    gram = matmul(transpose(q), q)
    check(max(abs(gram[i][j] - (1.0 if i == j else 0.0)) for i in range(width)
              for j in range(width)) < 1e-10,
          "the randomized range basis must be orthonormal")
    check(width == 9, "the default oversampling must add five sketch columns")
    thin = approximation_error(decaying, randomized_low_rank(decaying, 4, 0x7305, oversample=0))
    wide = approximation_error(decaying, randomized_low_rank(decaying, 4, 0x7305, oversample=5))
    bound4 = best_rank_k_error(decaying, 4)
    check(wide < thin, "oversampling must improve the range finder")
    check(wide <= 1.1 * bound4, "with oversampling the bound must hold tightly")
    check(thin > 1.2 * bound4, "without oversampling the sketch must miss part of the range")
    powered = approximation_error(decaying,
                                 randomized_low_rank(decaying, 4, 0x7305, power_iterations=1))
    check(powered <= 1.1 * bound4, "a power iteration must not degrade the approximation")

    # 6. boundary cases: rank zero, the full rank, and the zero matrix.
    zero_approximation = randomized_low_rank(decaying, 0, 0x7307)
    check(frobenius(zero_approximation) == 0.0, "rank zero must produce the zero matrix")
    check(abs(approximation_error(decaying, zero_approximation) - frobenius(decaying)) < 1e-9,
          "rank zero must leave the whole norm as error")
    full = randomized_low_rank(decaying, 10, 0x7307)
    check(approximation_error(decaying, full) < 1e-9, "the full rank must reproduce the matrix")
    zero_matrix = [[0.0] * 5 for _ in range(4)]
    check(randomized_low_rank(zero_matrix, 3, 0x7308) == [[0.0] * 5 for _ in range(4)],
          "the zero matrix must stay zero")
    check(best_rank_k_error(zero_matrix, 1) == 0.0, "the zero matrix has a zero tail")
    tall, seed = gaussian_matrix(9, 3, seed)
    check(randomized_low_rank(tall, 3, 0x7309) != [] and
          approximation_error(tall, randomized_low_rank(tall, 3, 0x7309)) < 1e-8,
          "a tall matrix must be handled")

    # 7. rejected inputs and determinism.
    expect_value_error(lambda: randomized_range(decaying, 99, 1), "a rank beyond the size")
    expect_value_error(lambda: randomized_range(decaying, -1, 1), "a negative rank")
    first = randomized_low_rank(decaying, 5, 0x7310)
    again = randomized_low_rank(decaying, 5, 0x7310)
    check(first == again, "the same seed must reproduce the same approximation")

    print(f"730 decay errors={[round(value, 6) for value in errors]}")
    print(f"730 best-rank bounds={[round(value, 6) for value in bounds]}")
    print(f"730 rank3 recovery={approximation_error(low, randomized_low_rank(low, 3, 0x7303)):.2e}")
    print("PASS")


if __name__ == "__main__":
    main()
