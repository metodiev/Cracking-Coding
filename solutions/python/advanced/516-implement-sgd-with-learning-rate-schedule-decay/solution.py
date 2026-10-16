"""Task 516 - Implement SGD with learning rate schedule (decay). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)

Stochastic gradient descent on a least-squares problem with a pluggable learning-rate schedule, plus
the closed forms the schedules are supposed to follow: constant, step decay
lr0 * drop^floor(t / every), exponential decay lr0 * gamma^t, inverse-time lr0 / (1 + decay * t),
cosine annealing lr_min + (lr_max - lr_min)(1 + cos(pi t / T)) / 2 and a linear warm-up followed by
cosine annealing. The model is ordinary least squares on a deterministic 40 x 3 design matrix: the
objective is (1/n) ||X theta - y||^2 with gradient (2/n) X^T (X theta - y), its Lipschitz constant is
L = 2 lambda_max(X^T X) / n (lambda_max by power iteration) and plain gradient descent therefore
diverges for lr > 2 / L = n / lambda_max - the precise bound the schedule experiments are built
around.

Documented checks: every schedule matches its closed form exactly at handpicked steps (step decay
1.0, 1.0, 0.5, 0.5, 0.25 at t = 0, 19, 20, 39, 40; exponential 0.99^t; inverse-time 1 / (1 + 0.01 t);
cosine hitting lr_max at t = 0, the midpoint at T/2 and lr_min at T; warm-up peaking exactly at the
warm-up length); all decaying schedules are non-increasing over 1000 steps while the warm-up is not
(it must rise first, and its peak is at the end of the warm-up); lambda_max is verified as a genuine
eigenvalue (residual <= 1e-9), against its Rayleigh quotient and inside the Gershgorin disc; the
normal-equation solution is checked to make the gradient exactly zero; with lr0 = 0.6 * (2 / L) and
inverse-time decay the average loss after 3000 steps sits within 1e-6 of the closed-form minimum and
the parameters within 5e-3 of theta*; a constant rate of 1.2 * (2 / L) makes the loss explode (at
least 10^6 times the initial value) while the same rate under inverse-time decay converges to the
minimum, which is the entire point of the schedule; a well-tuned constant rate also converges, so the
schedule is a convenience rather than a hack; and two identical runs produce identical parameters.
"""

import math

SAMPLES = 40
FEATURES = 3
STEPS = 3000
THETA_TRUE = [1.5, -2.0, 0.5]


def lcg(seed):
    return (seed * 75 + 74) % 65537


def make_problem(seed=2027, noise=0.01):
    """Design matrix, response y = X theta_true + tiny deterministic noise."""
    matrix = []
    for _ in range(SAMPLES):
        row = [1.0]
        for _ in range(FEATURES - 1):
            seed = lcg(seed)
            row.append((seed % 2001 - 1000) / 250.0)
        matrix.append(row)
    responses = []
    for row in matrix:
        seed = lcg(seed)
        responses.append(sum(a * b for a, b in zip(row, THETA_TRUE))
                         + (seed % 2001 - 1000) / 1000.0 * noise)
    return matrix, responses


def least_squares_loss(matrix, responses, theta):
    total = 0.0
    for row, target in zip(matrix, responses):
        residual = sum(a * b for a, b in zip(row, theta)) - target
        total += residual * residual
    return total / len(matrix)


def least_squares_gradient(matrix, responses, theta):
    gradient = [0.0] * len(theta)
    for row, target in zip(matrix, responses):
        residual = sum(a * b for a, b in zip(row, theta)) - target
        for k in range(len(theta)):
            gradient[k] += 2.0 * residual * row[k]
    scale = 1.0 / len(matrix)
    return [value * scale for value in gradient]


def normal_equations(matrix, responses):
    """Closed-form least squares by Gaussian elimination with partial pivoting."""
    d = len(matrix[0])
    gram = [[sum(row[i] * row[j] for row in matrix) for j in range(d)] for i in range(d)]
    rhs = [sum(row[i] * target for row, target in zip(matrix, responses)) for i in range(d)]
    augmented = [gram[i][:] + [rhs[i]] for i in range(d)]
    for column in range(d):
        pivot = max(range(column, d), key=lambda r: abs(augmented[r][column]))
        if abs(augmented[pivot][column]) <= 1e-15:
            raise ValueError("singular normal equations")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for row in range(d):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            augmented[row] = [value - factor * other
                              for value, other in zip(augmented[row], augmented[column])]
    return [augmented[i][d] for i in range(d)]


def power_iteration_max(matrix, iterations=200, seed=1):
    """lambda_max of a symmetric matrix by power iteration (deterministic start vector)."""
    d = len(matrix)
    vector = []
    for _ in range(d):
        seed = lcg(seed)
        vector.append((seed % 2001 - 1000) / 1000.0 + 0.5)
    length = math.sqrt(sum(v * v for v in vector))
    vector = [v / length for v in vector]
    value = 0.0
    for _ in range(iterations):
        product = [sum(matrix[i][j] * vector[j] for j in range(d)) for i in range(d)]
        length = math.sqrt(sum(v * v for v in product))
        vector = [v / length for v in product]
        value = sum(vector[i] * sum(matrix[i][j] * vector[j] for j in range(d))
                    for i in range(d))
    return value, vector


def gram_matrix(matrix):
    d = len(matrix[0])
    return [[sum(row[i] * row[j] for row in matrix) for j in range(d)] for i in range(d)]


# --- schedules -----------------------------------------------------------------------------

def constant_rate(rate):
    return lambda step: rate


def step_decay(rate, drop=0.5, every=20):
    return lambda step: rate * drop ** (step // every)


def exponential_decay(rate, gamma=0.99):
    return lambda step: rate * gamma ** step


def inverse_time(rate, decay=0.01):
    return lambda step: rate / (1.0 + decay * step)


def cosine_annealing(low, high, total):
    def schedule(step):
        clamped = min(max(step, 0), total)
        return low + 0.5 * (high - low) * (1.0 + math.cos(math.pi * clamped / total))
    return schedule


def warmup_cosine(low, high, warmup, total):
    cosine = cosine_annealing(low, high, total - warmup)

    def schedule(step):
        if step < warmup:
            return high * step / warmup
        return cosine(max(step - warmup, 0))
    return schedule


def sgd(matrix, responses, schedule, steps=STEPS, theta=None):
    """Full-gradient descent (deterministic data, so there is nothing to sample)."""
    parameters = list(theta) if theta is not None else [0.0] * len(matrix[0])
    history = [least_squares_loss(matrix, responses, parameters)]
    seen = [list(parameters)]
    for step in range(steps):
        rate = schedule(step)
        gradient = least_squares_gradient(matrix, responses, parameters)
        parameters = [value - rate * g for value, g in zip(parameters, gradient)]
        history.append(least_squares_loss(matrix, responses, parameters))
        seen.append(list(parameters))
    return parameters, history, seen


def main():
    matrix, responses = make_problem()
    gram = gram_matrix(matrix)
    lambda_max, eigenvector = power_iteration_max(gram)
    d = FEATURES

    # --- lambda_max is a real eigenvalue, above the average and inside Gershgorin. -----------
    residual = [sum(gram[i][j] * eigenvector[j] for j in range(d)) - lambda_max * eigenvector[i]
                for i in range(d)]
    assert math.sqrt(sum(v * v for v in residual)) <= 1e-9, "power iteration must reach an eigenpair"
    rayleigh = sum(eigenvector[i] * sum(gram[i][j] * eigenvector[j] for j in range(d))
                   for i in range(d))
    assert abs(rayleigh - lambda_max) <= 1e-12
    assert lambda_max >= sum(gram[i][i] for i in range(d)) / d - 1e-12, "at least the mean"
    gershgorin = max(gram[i][i] + sum(abs(gram[i][j]) for j in range(d) if j != i)
                     for i in range(d))
    assert lambda_max <= gershgorin + 1e-9, (lambda_max, gershgorin)
    lipschitz = 2.0 * lambda_max / len(matrix)
    stable_limit = 2.0 / lipschitz
    assert abs(stable_limit - len(matrix) / lambda_max) <= 1e-9
    assert stable_limit > 0.01, stable_limit

    # --- The closed-form optimum really is the optimum. -------------------------------------
    theta_star = normal_equations(matrix, responses)
    for value, want in zip(theta_star, THETA_TRUE):
        assert abs(value - want) <= 0.01, (theta_star, THETA_TRUE)
    closed_gradient = least_squares_gradient(matrix, responses, theta_star)
    assert max(abs(value) for value in closed_gradient) <= 1e-9, closed_gradient
    minimum_loss = least_squares_loss(matrix, responses, theta_star)
    assert 0.0 < minimum_loss < 1e-3, minimum_loss

    # --- Schedule closed forms. --------------------------------------------------------------
    step = step_decay(1.0)
    assert [step(t) for t in (0, 19, 20, 39, 40, 100)] == [1.0, 1.0, 0.5, 0.5, 0.25, 1.0 / 32.0]
    exponential = exponential_decay(1.0, gamma=0.99)
    for t in (0, 1, 5, 50, 200):
        assert exponential(t) == 0.99 ** t
    inverse = inverse_time(1.0, decay=0.01)
    for t in (0, 1, 2, 7, 250):
        assert inverse(t) == 1.0 / (1.0 + 0.01 * t)
    cosine = cosine_annealing(0.0, 1.0, 100)
    assert abs(cosine(0) - 1.0) <= 1e-15
    assert abs(cosine(50) - 0.5) <= 1e-15
    assert abs(cosine(100)) <= 1e-15 and abs(cosine(150)) <= 1e-15
    assert abs(cosine(25) - (0.5 + 0.5 * math.cos(math.pi / 4))) <= 1e-15
    assert constant_rate(0.7)(12345) == 0.7
    warm = warmup_cosine(0.0, 2.0, 10, 110)
    assert warm(0) == 0.0 and warm(5) == 1.0 and abs(warm(10) - 2.0) <= 1e-15
    assert abs(warm(60) - 1.0) <= 1e-15
    peak = max(range(0, 111), key=warm)
    assert peak == 10, ("the warm-up peaks exactly at its end", peak)

    # Decay schedules never increase; the warm-up must rise first.
    for schedule in (constant_rate(1.0), step, exponential, inverse, cosine, warm):
        values = [schedule(t) for t in range(0, 1000, 7)]
        assert min(values) >= 0.0
    for schedule in (step, exponential, inverse, cosine):
        values = [schedule(t) for t in range(1000)]
        assert all(b <= a + 1e-15 for a, b in zip(values, values[1:])), "must be non-increasing"
    warm_values = [warm(t) for t in range(111)]
    assert warm_values[0] < warm_values[10] and warm_values[10] > warm_values[110]

    # --- Inverse-time decay from a rate below the stability limit converges to the minimum. --
    rate0 = 0.6 * stable_limit
    schedule = inverse_time(rate0, decay=0.01)
    parameters, history, trajectory = sgd(matrix, responses, schedule)
    assert abs(history[-1] - minimum_loss) <= 1e-6, (history[-1], minimum_loss)
    assert max(abs(a - b) for a, b in zip(parameters, theta_star)) <= 5e-3, parameters
    assert history[-1] < history[0] / 100.0

    # --- The same rate kept constant above the limit explodes; decayed, it converges. --------
    unstable_rate = 1.2 * stable_limit
    assert unstable_rate > stable_limit
    burst_parameters, burst_history, _ = sgd(matrix, responses, constant_rate(unstable_rate),
                                             steps=300)
    assert burst_history[-1] > 1e6 * burst_history[0], burst_history[-1]
    tame_parameters, tame_history, _ = sgd(matrix, responses,
                                           inverse_time(unstable_rate, decay=0.05), steps=3000)
    assert abs(tame_history[-1] - minimum_loss) <= 1e-5, tame_history[-1]
    # The constant rate grows monotonically once it has passed the stable region ...
    assert all(second >= first for first, second in
               zip(burst_history[2:], burst_history[3:])), "the burst keeps growing"
    # ... while the decayed run overshoots first (its loss grows by more than 50%) and then settles.
    assert max(tame_history[:20]) > 1.5 * tame_history[0], max(tame_history[:20])
    assert burst_history[:2] == tame_history[:2], "the first step uses the same rate"
    assert burst_history[2] != tame_history[2], "then the decay takes over"
    assert burst_history[-1] / tame_history[-1] > 1e6, "totally different destinations"

    # --- A well-tuned constant rate converges too, so the schedule is a convenience. --------
    reference, reference_history, _ = sgd(matrix, responses, constant_rate(0.9 * stable_limit))
    assert abs(reference_history[-1] - minimum_loss) <= 1e-5
    assert max(abs(a - b) for a, b in zip(reference, theta_star)) <= 5e-3
    assert history[-1] <= reference_history[-1] + 1e-4

    # --- Determinism. -----------------------------------------------------------------------
    again, again_history, again_seen = sgd(matrix, responses, inverse_time(rate0, decay=0.01))
    assert again == parameters and again_history == history
    assert again_seen == trajectory, "the whole trajectory must be reproducible"
    assert len(history) == STEPS + 1


if __name__ == "__main__":
    main()
    print("PASS")
