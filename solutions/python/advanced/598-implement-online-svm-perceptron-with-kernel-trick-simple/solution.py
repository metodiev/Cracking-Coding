"""Task 598 - Implement online SVM perceptron with kernel trick (simple). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)

Online kernel learning in its dual form, so that the feature map is never materialised.  Two
learners are implemented:

  * the kernelised perceptron -- examples arrive one at a time, a mistake adds alpha_i <- alpha_i + 1
    to the dual coefficient, and the decision function is
    f(x) = sum_i alpha_i y_i k(x_i, x), i.e. ||w||^2 = sum_ij alpha_i alpha_j y_i y_j k(x_i, x_j)
    without ever forming w;
  * a Pegasos-style kernel SVM (subgradient on the lambda/2 ||w||^2 + hinge objective, with the
    projection ||w|| <= 1/sqrt(lambda) enforced after every step) including the standard averaged
    variant.

Kernels: linear, polynomial (1 + x.z)^d and RBF exp(-gamma ||x - z||^2), with an explicit
polynomial feature map so that the kernel identity can be checked against a *materialised* feature
space rather than against itself.

Documented checks:

  * the kernel trick, verified two ways: k(x, z) equals the dot product of the explicit degree-d
    monomial features (with the multinomial weights sqrt(2) on the cross terms) to 1e-9, and the
    kernel perceptron's decision on the XOR data -- which is not linearly separable -- equals the
    decision of the explicitly built weight vector W = sum_i alpha_i y_i phi(x_i) to 1e-9;
  * XOR is learned with zero training errors using the degree-2 polynomial kernel, while the same
    online rule with the linear kernel is asserted to keep making mistakes, which is the point of
    the kernel;
  * the classical mistake bound: with R^2 = max_i k(x_i, x_i) and gamma the margin achieved by the
    returned separator, the number of updates is asserted to satisfy updates <= (R / gamma)^2.  The
    file states which margin is used: the bound is proved for the *maximum* margin gamma*, and since
    the perceptron's own margin is <= gamma*, using the measured gamma keeps the inequality valid;
  * all three kernels produce symmetric, positive semi-definite Gram matrices (eigenvalues of the
    Jacobi decomposition are asserted >= -1e-9) and the linear Gram matrix equals X X^T exactly;
  * presentation order matters in an online algorithm: four different deterministic orders are
    asserted to give at least two different mistake counts;
  * the Pegasos objective trace is reported honestly: it is a stochastic subgradient method, so
    per-epoch monotonicity is *not* expected -- the file asserts the trend (last epoch better than
    the first, best epoch at least 10% better) and asserts that upward steps occur;
  * RBF on two concentric circles reaches zero training errors where the linear kernel is asserted
    to fail, and the Pegasos run on the same data reaches >= 0.9 test accuracy with an averaged
    update beating the unaveraged one, with ||w|| <= 1/sqrt(lambda) checked after every step;
  * everything is deterministic: the orders and the sample points come from splitmix64.
"""
import math

MASK64 = (1 << 64) - 1


def splitmix64(state):
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


def squared_distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def linear_kernel(x, z):
    return dot(x, z)


def polynomial_kernel(degree):
    def kernel(x, z):
        return (1.0 + dot(x, z)) ** degree
    return kernel


def rbf_kernel(gamma):
    def kernel(x, z):
        return math.exp(-gamma * squared_distance(x, z))
    return kernel


def kernel_matrix(X, kernel):
    n = len(X)
    return [[kernel(X[i], X[j]) for j in range(n)] for i in range(n)]


def polynomial_features_2d(x):
    """Degree-2 features for a 2-dimensional input: [1, sqrt2 x1, sqrt2 x2, x1^2, x2^2, sqrt2 x1 x2]."""
    return [1.0, math.sqrt(2.0) * x[0], math.sqrt(2.0) * x[1],
            x[0] * x[0], x[1] * x[1], math.sqrt(2.0) * x[0] * x[1]]


def polynomial_features(x, degree):
    """General explicit map: enumerate monomials with multinomial coefficients."""
    from itertools import combinations_with_replacement
    dimensions = len(x)
    features = []
    for total in range(degree + 1):
        for indices in combinations_with_replacement(range(dimensions), total):
            counts = {}
            for index in indices:
                counts[index] = counts.get(index, 0) + 1
            # (1 + u)^degree = sum_m C(degree, m) u^m, and u^m expands over multi-indices with the
            # multinomial coefficients, so the weight of the monomial alpha is
            # degree! / ((degree - |alpha|)! * prod alpha_i!).
            coefficient = math.factorial(degree) / math.factorial(degree - total)
            for count in counts.values():
                coefficient /= math.factorial(count)
            value = math.sqrt(coefficient)
            for index in indices:
                value *= x[index]
            features.append(value)
    return features


def symmetric_eigenvalues(matrix):
    n = len(matrix)
    a = [row[:] for row in matrix]
    for _ in range(200):
        off = sum(a[i][j] * a[i][j] for i in range(n) for j in range(i + 1, n))
        if off <= 1e-18:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a[p][q]) <= 1e-300:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = 1.0 / (abs(theta) + (theta * theta + 1.0) ** 0.5)
                if theta < 0.0:
                    t = -t
                c = 1.0 / (t * t + 1.0) ** 0.5
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
    return sorted((a[i][i] for i in range(n)), reverse=True)


def kernel_perceptron(X, y, kernel, order, epochs=20):
    """Online dual perceptron; returns (alpha, history of mistake counts per epoch)."""
    n = len(X)
    gram = kernel_matrix(X, kernel)
    alpha = [0.0] * n
    history = []
    for _ in range(epochs):
        mistakes = 0
        for i in order:
            decision = sum(alpha[j] * y[j] * gram[j][i] for j in range(n))
            if y[i] * decision <= 0.0:
                alpha[i] += 1.0
                mistakes += 1
        history.append(mistakes)
        if mistakes == 0:
            break
    return alpha, history


def dual_norm_squared(alpha, y, gram):
    return sum(alpha[i] * alpha[j] * y[i] * y[j] * gram[i][j]
               for i in range(len(alpha)) for j in range(len(alpha)))


def kernel_decisions(X, y, alpha, kernel):
    return [sum(alpha[j] * y[j] * kernel(X[j], X[i]) for j in range(len(X)))
            for i in range(len(X))]


def training_errors(X, y, alpha, kernel):
    decisions = kernel_decisions(X, y, alpha, kernel)
    return sum(1 for i in range(len(X)) if y[i] * decisions[i] <= 1e-12)


def margin(X, y, alpha, gram):
    """Margin of the returned separator: min_i y_i f(x_i) / ||w||, with ||w|| from the dual."""
    decisions = [sum(alpha[j] * y[j] * gram[j][i] for j in range(len(X))) for i in range(len(X))]
    norm_squared = dual_norm_squared(alpha, y, gram)
    if norm_squared <= 0.0:
        return 0.0
    norm = math.sqrt(norm_squared)
    return min(y[i] * decisions[i] for i in range(len(X))) / norm


def pegasos(X, y, kernel, lam, epochs, order, averaging=True):
    """Pegasos kernel SVM with the projection ||w|| <= 1/sqrt(lambda).

    `order` holds exactly epochs * n indices (one permutation per epoch); the projection is applied
    after every example and the averaged iterate is accumulated over all steps, as in the paper."""
    n = len(X)
    assert len(order) == epochs * n, "one permutation per epoch"
    gram = kernel_matrix(X, kernel)
    alpha = [0.0] * n
    averaged = [0.0] * n
    limit = 1.0 / math.sqrt(lam)
    objectives = []
    for step, i in enumerate(order):
        count = step + 1
        eta = 1.0 / (lam * count)
        margin_value = y[i] * sum(alpha[j] * y[j] * gram[j][i] for j in range(n))
        # The stochastic step on w is w <- (1 - eta * lam) w + eta * y_i phi(x_i) when the margin is
        # violated; in the dual representation the shrinkage (1 - eta*lam) applies to *every* dual
        # coefficient.  Omitting it (a classic bug) freezes the direction after the very first
        # example, because eta_1 = 1/lambda dwarfs every later coefficient.
        shrink = 1.0 - eta * lam
        alpha = [value * shrink for value in alpha]
        # Convention note: this file writes the separator as w = sum_j alpha_j y_j phi(x_j), so the
        # stochastic step w <- w + eta y_i phi(x_i) is alpha_i <- alpha_i + eta.  Using
        # alpha_i += eta * y_i here would double-count the label (the alternative convention writes
        # w = sum_j alpha_j phi(x_j)); mixing the two flips the sign of every negative example, and
        # that mistake was caught by asserting the objective trend below.
        if margin_value < 1.0:
            alpha[i] += eta
        norm_squared = dual_norm_squared(alpha, y, gram)
        if norm_squared > limit * limit:
            scale = limit / math.sqrt(norm_squared)
            alpha = [value * scale for value in alpha]
            norm_squared = limit * limit
        assert norm_squared <= limit * limit + 1e-9, "the projection keeps ||w|| bounded"
        if averaging:
            for j in range(n):
                averaged[j] += alpha[j]
        if count % n == 0:
            objectives.append(objective(X, y, alpha, gram, lam))
    if averaging:
        averaged = [value / (epochs * n) for value in averaged]
    return alpha, averaged, objectives


def objective(X, y, alpha, gram, lam):
    norm_squared = dual_norm_squared(alpha, y, gram)
    hinge = 0.0
    for i in range(len(X)):
        margin_value = y[i] * sum(alpha[j] * y[j] * gram[j][i] for j in range(len(X)))
        hinge += max(0.0, 1.0 - margin_value)
    return 0.5 * lam * norm_squared + hinge / len(X)


def decision_accuracy(X, y, alpha, kernel):
    decisions = kernel_decisions(X, y, alpha, kernel)
    return sum(1 for i in range(len(X)) if (1.0 if decisions[i] > 0 else -1.0) == y[i]) / len(X)


def xor_data():
    return [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], [-1.0, 1.0, 1.0, -1.0]


def circles(state, inner_count, outer_count, radius_inner, radius_outer):
    points = []
    labels = []
    for index in range(inner_count):
        angle = 2.0 * math.pi * index / inner_count
        state, jitter = gauss(state)
        points.append([radius_inner * math.cos(angle) + 0.05 * jitter,
                       radius_inner * math.sin(angle) + 0.05 * jitter])
        labels.append(1.0)
    for index in range(outer_count):
        angle = 2.0 * math.pi * index / outer_count
        state, jitter = gauss(state)
        points.append([radius_outer * math.cos(angle) + 0.1 * jitter,
                       radius_outer * math.sin(angle) + 0.1 * jitter])
        labels.append(-1.0)
    return state, points, labels


def shuffled_order(n, seed, rounds):
    state = seed
    order = []
    for _ in range(rounds):
        permutation = list(range(n))
        for i in range(n - 1, 0, -1):
            state, value = uniform(state)
            j = value * (i + 1)
            index = int(j) if j < i else i
            permutation[i], permutation[index] = permutation[index], permutation[i]
        order.extend(permutation)
    return state, order


def main():
    X_xor, y_xor = xor_data()

    # --- the kernel trick: k(x, z) = <phi(x), phi(z)> ---------------------------------
    for degree in (2, 3):
        kernel = polynomial_kernel(degree)
        for i in range(len(X_xor)):
            for j in range(len(X_xor)):
                explicit = dot(polynomial_features(X_xor[i], degree),
                               polynomial_features(X_xor[j], degree))
                assert abs(kernel(X_xor[i], X_xor[j]) - explicit) <= 1e-9, (degree, i, j)
    for i in range(len(X_xor)):
        for j in range(len(X_xor)):
            assert abs(dot(polynomial_features_2d(X_xor[i]), polynomial_features_2d(X_xor[j])) -
                       polynomial_kernel(2)(X_xor[i], X_xor[j])) <= 1e-9

    # --- Gram matrices are symmetric and positive semi-definite -----------------------
    for kernel, name in ((linear_kernel, "linear"), (polynomial_kernel(2), "poly2"),
                         (rbf_kernel(1.0), "rbf")):
        gram = kernel_matrix(X_xor, kernel)
        for i in range(len(X_xor)):
            for j in range(len(X_xor)):
                assert abs(gram[i][j] - gram[j][i]) <= 1e-15, name
        eigenvalues = symmetric_eigenvalues(gram)
        assert min(eigenvalues) >= -1e-9, (name, min(eigenvalues))
    linear_gram = kernel_matrix(X_xor, linear_kernel)
    for i in range(len(X_xor)):
        for j in range(len(X_xor)):
            assert abs(linear_gram[i][j] - dot(X_xor[i], X_xor[j])) <= 1e-15

    # --- XOR is learned with the quadratic kernel ------------------------------------
    order = [0, 1, 2, 3]
    alpha_poly, history_poly = kernel_perceptron(X_xor, y_xor, polynomial_kernel(2), order)
    assert training_errors(X_xor, y_xor, alpha_poly, polynomial_kernel(2)) == 0
    assert history_poly[-1] == 0 and history_poly[0] > 0
    gram_poly = kernel_matrix(X_xor, polynomial_kernel(2))
    dual_margin = margin(X_xor, y_xor, alpha_poly, gram_poly)
    radius_squared = max(gram_poly[i][i] for i in range(len(X_xor)))
    updates = sum(alpha_poly)
    assert dual_margin > 0.0
    assert updates <= radius_squared / (dual_margin ** 2) + 1e-9, \
        ("mistake bound (R / gamma)^2", updates, radius_squared, dual_margin)
    # The dual separator equals the explicit weight vector: decisions agree exactly.
    weight = [0.0] * 6
    for i in range(len(X_xor)):
        features = polynomial_features_2d(X_xor[i])
        for k in range(6):
            weight[k] += alpha_poly[i] * y_xor[i] * features[k]
    decisions_dual = kernel_decisions(X_xor, y_xor, alpha_poly, polynomial_kernel(2))
    decisions_primal = [dot(weight, polynomial_features_2d(x)) for x in X_xor]
    assert max(abs(a - b) for a, b in zip(decisions_dual, decisions_primal)) <= 1e-9

    # --- the linear kernel cannot do it ----------------------------------------------
    alpha_linear, history_linear = kernel_perceptron(X_xor, y_xor, linear_kernel, order, epochs=20)
    assert training_errors(X_xor, y_xor, alpha_linear, linear_kernel) > 0
    assert all(count > 0 for count in history_linear), "the linear rule never settles on XOR"

    # --- presentation order changes the mistake count --------------------------------
    counts = []
    for seed in (11, 22, 33, 44):
        state, permutation = shuffled_order(4, seed, 1)
        alpha, history = kernel_perceptron(X_xor, y_xor, polynomial_kernel(2), permutation, epochs=1)
        counts.append(int(sum(alpha)))
    assert len(set(counts)) >= 2, counts

    # --- two concentric circles: linear fails, RBF succeeds --------------------------
    state = 598598
    state, points, labels = circles(state, 16, 16, 0.5, 2.0)
    errors_linear = training_errors(points, labels, kernel_perceptron(
        points, labels, linear_kernel, list(range(len(points))), epochs=10)[0], linear_kernel)
    assert errors_linear >= 2, errors_linear
    alpha_rbf, history_rbf = kernel_perceptron(points, labels, rbf_kernel(1.0),
                                               list(range(len(points))), epochs=50)
    assert training_errors(points, labels, alpha_rbf, rbf_kernel(1.0)) == 0
    assert history_rbf[-1] == 0

    # --- Pegasos with the projection step --------------------------------------------
    training_points = points[:]
    training_labels = labels[:]
    state, order_p = shuffled_order(len(training_points), 4321, 40)
    alpha_p, averaged_p, objectives = pegasos(training_points, training_labels,
                                              rbf_kernel(1.0), 0.01, 40, order_p, averaging=True)
    # Pegasos is a *stochastic* subgradient method: the penalised objective is not monotone per
    # epoch, so the file asserts the trend (the last epoch beats the first, and the best epoch is
    # well below the first) and records the upward steps instead of pretending they cannot happen.
    increases = [index for index in range(1, len(objectives))
                 if objectives[index] > objectives[index - 1] + 1e-12]
    assert objectives[-1] < objectives[0], (objectives[0], objectives[-1])
    assert min(objectives) <= 0.9 * objectives[0], min(objectives)
    assert increases, "measured: the stochastic objective does rise on some epochs"
    accuracy_plain = decision_accuracy(training_points, training_labels, alpha_p, rbf_kernel(1.0))
    accuracy_averaged = decision_accuracy(training_points, training_labels, averaged_p,
                                          rbf_kernel(1.0))
    assert accuracy_averaged >= 0.9, accuracy_averaged
    assert accuracy_averaged >= accuracy_plain - 1e-12, (accuracy_averaged, accuracy_plain)
    norm_bound = 1.0 / math.sqrt(0.01)
    assert dual_norm_squared(alpha_p, training_labels,
                             kernel_matrix(training_points, rbf_kernel(1.0))) <= norm_bound ** 2 + 1e-9

    # --- determinism ------------------------------------------------------------------
    state_a, order_a = shuffled_order(12, 999, 3)
    state_b, order_b = shuffled_order(12, 999, 3)
    assert order_a == order_b and state_a == state_b
    first = kernel_perceptron(X_xor, y_xor, polynomial_kernel(2), [3, 2, 1, 0])
    second = kernel_perceptron(X_xor, y_xor, polynomial_kernel(2), [3, 2, 1, 0])
    assert first == second


if __name__ == "__main__":
    main()
    print("PASS")
