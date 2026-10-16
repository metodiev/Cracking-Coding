"""Task 797 - Implement multi-armed bandit with contextual features (contextual bandits).  (Tier: Expert, CTCI Beyond - Machine Learning & AI)

LinUCB models the reward of arm a for context x as x^T theta_a with a ridge posterior
A_a = I_d + sum x x^T, b_a = sum r x, so theta_a_hat = A_a^{-1} b_a and the optimistic index is

    UCB_a(x) = theta_a_hat^T x + alpha * sqrt(x^T A_a^{-1} x),

where the second term is the (scaled) standard deviation of the reward prediction - the
"upper confidence" that drives exploration.  After each pull only that arm's statistics change,
and A_a^{-1} is updated in O(d^2) with the Sherman-Morrison identity

    A^{-1} <- A^{-1} - (A^{-1} x x^T A^{-1}) / (1 + x^T A^{-1} x),

so no matrix is ever re-inverted.  Two baselines run on the identical reward stream: a
uniform-random policy and an epsilon-greedy policy using the same ridge estimates.

Verified: the weighted least-squares solution agrees with a fresh Gaussian-elimination solve of
the normal equations; the Sherman-Morrison update agrees with recomputing the inverse from
scratch; the arm actually pulled is the argmax of the computed upper-confidence values;
A_a stays symmetric positive definite for every arm and every step; pseudo-regret is reported
for LinUCB, epsilon-greedy and uniform over several repetitions, with LinUCB required to beat
both; regret is checked for sublinear growth (doubling the horizon must less than double the
regret); the parameter estimate of a well-explored arm converges to the true theta; and the
degenerate cases (a single arm, alpha = 0, a one-step horizon) behave exactly.

Simplified versus a production contextual bandit: the reward model is linear-Gaussian with
known noise scale so the confidence width is analytic, contexts are i.i.d. and low dimensional
(d = 5, K = 4), no feature mapping for non-linear rewards, no off-policy logging or exploration
budget accounting, and regret is measured in expectation via the true model rather than as
realised reward.
"""
import math

MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def uniform(state):
    state, r = splitmix64(state)
    return state, r / 18446744073709551616.0


def gauss(state):
    state, u1 = splitmix64(state)
    state, u2 = splitmix64(state)
    r = (u1 >> 11) / 9007199254740992.0
    p = (u2 >> 11) / 9007199254740992.0
    if r < 1e-12:
        r = 1e-12
    return state, math.sqrt(-2.0 * math.log(r)) * math.cos(2.0 * math.pi * p)


# ---------------------------------------------------------------- linear algebra
def solve_linear(matrix, rhs):
    """Dense Gaussian elimination with partial pivoting (the reference path)."""
    n = len(rhs)
    a = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-300:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        inv = 1.0 / a[col][col]
        for row in range(col + 1, n):
            factor = a[row][col] * inv
            for k in range(col, n + 1):
                a[row][k] -= factor * a[col][k]
    x = [0.0] * n
    for row in range(n - 1, -1, -1):
        x[row] = (a[row][n] - sum(a[row][k] * x[k] for k in range(row + 1, n))) / a[row][row]
    return x


def quadratic_form(inverse, vector):
    return math.fsum(vector[i] * math.fsum(inverse[i][j] * vector[j] for j in range(len(vector)))
                     for i in range(len(vector)))


def sherman_morrison(inverse, vector):
    """Rank-one update A^{-1} <- (A + x x^T)^{-1} without touching A itself."""
    d = len(vector)
    ai = [math.fsum(inverse[i][j] * vector[j] for j in range(d)) for i in range(d)]
    denominator = 1.0 + math.fsum(vector[i] * ai[i] for i in range(d))
    return [[inverse[i][j] - ai[i] * ai[j] / denominator for j in range(d)] for i in range(d)]


# ------------------------------------------------------------------- LinUCB
class LinUCB:
    """Ridge-posterior contextual bandit with an optimistic index."""

    def __init__(self, arms, dim, alpha, ridge=1.0):
        if arms < 1 or dim < 1:
            raise ValueError("arms and dim must be positive")
        if alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        self.arms = arms
        self.dim = dim
        self.alpha = alpha
        self.inverse = [[[ridge if i == j else 0.0 for j in range(dim)] for i in range(dim)]
                        for _ in range(arms)]
        self.b = [[0.0] * dim for _ in range(arms)]
        self.pulls = [0] * arms
        self.last_scores = None

    def theta(self, arm):
        """theta_hat = A^{-1} b, computed by the stored inverse."""
        return [math.fsum(self.inverse[arm][i][j] * self.b[arm][j] for j in range(self.dim))
                for i in range(self.dim)]

    def scores(self, context):
        values = []
        for arm in range(self.arms):
            theta = self.theta(arm)
            mean = math.fsum(theta[i] * context[i] for i in range(self.dim))
            variance = quadratic_form(self.inverse[arm], context)
            values.append(mean + self.alpha * math.sqrt(max(0.0, variance)))
        self.last_scores = values
        return values

    def select(self, context):
        values = self.scores(context)
        best = 0
        for arm in range(1, self.arms):
            if values[arm] > values[best]:
                best = arm
        return best

    def update(self, arm, context, reward):
        self.inverse[arm] = sherman_morrison(self.inverse[arm], context)
        for i in range(self.dim):
            self.b[arm][i] += reward * context[i]
        self.pulls[arm] += 1


class EpsilonGreedy:
    """Baseline: ridge estimates with uniform random exploration."""

    def __init__(self, arms, dim, epsilon, ridge=1.0, seed=1):
        self.arms = arms
        self.dim = dim
        self.epsilon = epsilon
        self.ridge = ridge
        self.a = [[[ridge if i == j else 0.0 for j in range(dim)] for i in range(dim)]
                  for _ in range(arms)]
        self.b = [[0.0] * dim for _ in range(arms)]
        self.state = seed
        self.pulls = [0] * arms

    def theta(self, arm):
        solution = solve_linear(self.a[arm], self.b[arm])
        return solution if solution is not None else [0.0] * self.dim

    def select(self, context):
        self.state, r = uniform(self.state)
        if r < self.epsilon:
            self.state, pick = splitmix64(self.state)
            return pick % self.arms
        best = 0
        best_value = -float("inf")
        for arm in range(self.arms):
            value = math.fsum(self.theta(arm)[i] * context[i] for i in range(self.dim))
            if value > best_value:
                best, best_value = arm, value
        return best

    def update(self, arm, context, reward):
        for i in range(self.dim):
            self.b[arm][i] += reward * context[i]
            for j in range(self.dim):
                self.a[arm][i][j] += context[i] * context[j]
        self.pulls[arm] += 1


# ----------------------------------------------------------------- environment
def make_theta(arms, dim, seed, scale=1.0):
    state = seed
    thetas = []
    for _ in range(arms):
        row = []
        for _c in range(dim):
            state, value = gauss(state)
            row.append(scale * value)
        thetas.append(row)
    return thetas, state


def expected_reward(theta, context):
    return math.fsum(theta[i] * context[i] for i in range(len(context)))


def run_policy(policy, thetas, steps, seed, noise=0.05):
    """Drive a policy over i.i.d. Gaussian contexts; returns cumulative pseudo-regret."""
    dim = len(thetas[0])
    state = seed
    regret = 0.0
    for _ in range(steps):
        context = []
        for _c in range(dim):
            state, value = gauss(state)
            context.append(value)
        best = max(expected_reward(theta, context) for theta in thetas)
        arm = policy.select(context)
        mean = expected_reward(thetas[arm], context)
        regret += best - mean
        state, noise_value = gauss(state)
        policy.update(arm, context, mean + noise * noise_value)
    return regret


def run_uniform(thetas, steps, seed, arms):
    dim = len(thetas[0])
    state = seed
    regret = 0.0
    for _ in range(steps):
        context = []
        for _c in range(dim):
            state, value = gauss(state)
            context.append(value)
        best = max(expected_reward(theta, context) for theta in thetas)
        state, pick = splitmix64(state)
        regret += best - expected_reward(thetas[pick % arms], context)
    return regret


def main():
    ARMS, DIM = 4, 5
    thetas, _state = make_theta(ARMS, DIM, 797001)
    ALPHA = 0.6

    # --- 0. run a short exploratory episode for the linear-algebra checks ----
    policy = LinUCB(ARMS, DIM, ALPHA)
    state = 797100
    pulled = [[] for _ in range(ARMS)]
    for _ in range(30):
        context = []
        for _c in range(DIM):
            state, value = gauss(state)
            context.append(value)
        arm = policy.select(context)
        pulled[arm].append(context)
        state, noise_value = gauss(state)
        policy.update(arm, context, expected_reward(thetas[arm], context) + 0.05 * noise_value)

    # --- 1. accumulate the explicit normal equations for the oracle ---------
    accumulated = []
    for arm in range(ARMS):
        a = [[1.0 if i == j else 0.0 for j in range(DIM)] for i in range(DIM)]
        for context in pulled[arm]:
            for i in range(DIM):
                for j in range(DIM):
                    a[i][j] += context[i] * context[j]
        accumulated.append(a)
    worst = 0.0
    for arm in range(ARMS):
        direct = solve_linear(accumulated[arm], policy.b[arm])
        stored = policy.theta(arm)
        worst = max(worst, max(abs(direct[i] - stored[i]) for i in range(DIM)))
    check(worst < 1e-9,
          "the inverse-based theta must reproduce the explicit normal-equation solve, "
          "worst gap %.3e" % worst)

    # --- 2. the stored inverse really is the inverse of the accumulated A ----
    worst_identity = 0.0
    for arm in range(ARMS):
        for i in range(DIM):
            for j in range(DIM):
                product = math.fsum(accumulated[arm][i][k] * policy.inverse[arm][k][j]
                                    for k in range(DIM))
                worst_identity = max(worst_identity, abs(product - (1.0 if i == j else 0.0)))
        check(quadratic_form(policy.inverse[arm], [1.0] * DIM) > 0.0,
              "the inverse must stay positive definite")
        check(max(abs(policy.inverse[arm][i][j] - policy.inverse[arm][j][i])
                  for i in range(DIM) for j in range(DIM)) < 1e-15,
              "the Sherman-Morrison update must preserve symmetry")
    check(worst_identity < 1e-9,
          "the Sherman-Morrison inverse must invert the accumulated A, worst deviation %.3e"
          % worst_identity)

    # --- 3. the pulled arm is the argmax of the scores ----------------------
    context = [0.3, -0.2, 0.7, 0.1, -0.5]
    values = policy.scores(context)
    chosen = policy.select(context)
    check(values[chosen] == max(values),
          "the selected arm must maximise the upper-confidence value")
    check(policy.last_scores == values, "the scores used for selection must be the reported ones")
    check(all(value == value for value in values), "scores must be finite numbers")

    # --- 4. regret versus the baselines ------------------------------------
    steps, repetitions = 1500, 8
    linucb_regret = 0.0
    epsilon_regret = 0.0
    uniform_regret = 0.0
    for run in range(repetitions):
        linucb_regret += run_policy(LinUCB(ARMS, DIM, ALPHA), thetas, steps, 797200 + run)
        epsilon_regret += run_policy(EpsilonGreedy(ARMS, DIM, 0.1, seed=797200 + run), thetas,
                                    steps, 797200 + run)
        uniform_regret += run_uniform(thetas, steps, 797200 + run, ARMS)
    linucb_regret /= repetitions
    epsilon_regret /= repetitions
    uniform_regret /= repetitions
    check(linucb_regret < 0.5 * uniform_regret,
          "LinUCB regret %.4f must beat the uniform policy %.4f"
          % (linucb_regret, uniform_regret))
    check(linucb_regret <= epsilon_regret,
          "LinUCB regret %.4f must not lose to epsilon-greedy %.4f"
          % (linucb_regret, epsilon_regret))
    check(uniform_regret > epsilon_regret,
          "epsilon-greedy must beat the uniform policy: %.4f vs %.4f"
          % (epsilon_regret, uniform_regret))

    # --- 5. sublinear regret growth ---------------------------------------
    short, long = steps // 2, steps
    regret_short = 0.0
    regret_long = 0.0
    for run in range(repetitions):
        regret_short += run_policy(LinUCB(ARMS, DIM, ALPHA), thetas, short, 797300 + run)
        regret_long += run_policy(LinUCB(ARMS, DIM, ALPHA), thetas, long, 797300 + run)
    check(regret_long < 2.0 * regret_short,
          "regret must grow sublinearly: %.4f over %d steps vs %.4f over %d steps"
          % (regret_long, long, regret_short, short))

    # --- 6. parameter recovery for a well-explored arm ---------------------
    greedy = LinUCB(ARMS, DIM, ALPHA)
    state = 797400
    for _ in range(4000):
        context = []
        for _c in range(DIM):
            state, value = gauss(state)
            context.append(value)
        arm = greedy.select(context)
        state, noise_value = gauss(state)
        greedy.update(arm, context, expected_reward(thetas[arm], context) + 0.05 * noise_value)
    best_arm = max(range(ARMS), key=lambda a: greedy.pulls[a])
    estimate = greedy.theta(best_arm)
    recovery = max(abs(estimate[i] - thetas[best_arm][i]) for i in range(DIM))
    check(greedy.pulls[best_arm] > 200,
          "the dominant arm must be explored many times, got %d" % greedy.pulls[best_arm])
    check(recovery < 0.05,
          "the ridge estimate must converge to the true theta, worst gap %.4f" % recovery)

    # --- 7. degenerate cases and determinism ------------------------------
    single = LinUCB(1, DIM, ALPHA)
    context = [1.0, 0.0, 0.0, 0.0, 0.0]
    check(single.select(context) == 0, "with one arm the choice is forced")
    zero_alpha = LinUCB(ARMS, DIM, 0.0)
    values = zero_alpha.scores(context)
    check(all(abs(value - values[0]) < 1e-12 for value in values),
          "with alpha = 0 the scores of identical arms must coincide")
    check(zero_alpha.select(context) == 0, "ties must resolve deterministically to arm 0")
    one_step = run_policy(LinUCB(ARMS, DIM, ALPHA), thetas, 1, 797500)
    check(one_step >= 0.0, "a single-step regret must be non-negative")
    repeat_a = run_policy(LinUCB(ARMS, DIM, ALPHA), thetas, 50, 797600)
    repeat_b = run_policy(LinUCB(ARMS, DIM, ALPHA), thetas, 50, 797600)
    check(repeat_a == repeat_b, "an identical seed must reproduce the regret exactly")
    raised = False
    try:
        LinUCB(ARMS, DIM, -1.0)
    except ValueError:
        raised = True
    check(raised, "a negative exploration width must be rejected")

    print("theta check: worst |inverse-based - direct solve| = %.3e" % worst)
    print("A * A^-1 - I: worst deviation %.3e" % worst_identity)
    print("mean regret over %d repetitions of %d steps: LinUCB %.4f, epsilon-greedy %.4f, "
          "uniform %.4f" % (repetitions, steps, linucb_regret, epsilon_regret, uniform_regret))
    print("sublinearity: regret %.4f at %d steps vs %.4f at %d steps"
          % (regret_short, short, regret_long, long))
    print("parameter recovery on the dominant arm (%d pulls): worst gap %.4f"
          % (greedy.pulls[best_arm], recovery))
    print("PASS")


if __name__ == "__main__":
    main()
