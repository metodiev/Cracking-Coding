"""Task 896 - Implement joint optimization of storage layout + access patterns using reinforcement learning.  (Tier: Master, CTCI Beyond - Distributed Systems & Storage)

The problem is a real storage decision: a column store must choose which columns share a physical
block *and* when it is worth rewriting those blocks, because every reorganisation costs I/O while
every badly grouped query pays a block-open for each block it touches.  Layout and access pattern
therefore have to be optimised together - clustering the columns a query co-accesses is only
useful if the pattern repeats often enough to amortise the rewrite.

The instance is a workload cycle of `P` queries over `C` columns, each column with a byte width,
and a candidate set of layouts (partitions of the columns into blocks).  A query under a layout
costs one unit per block it touches plus the bytes of the columns it reads, and switching layouts
costs `penalty` per column whose block membership changed.  That is modelled exactly as a finite
horizon Markov decision process: the state is (position in the cycle, current layout), an action
picks the layout for the next query, and the reward is the negated query plus reorganisation
cost.  A backward dynamic program solves it exactly, an exhaustive enumeration of action
sequences is used to confirm the dynamic program on a tiny instance, and tabular Q-learning is
then asked to discover the same policy with no knowledge of the model at all.

Verified: the dynamic program satisfies the Bellman optimality equation at every state and
matches exhaustive action-sequence enumeration bit for bit, the transition and reward functions
are deterministic and reorg costs are accounted exactly, Q-learning's visits-based learning rate
converges so that its greedy roll-out matches the dynamic program's optimal cycle cost, the
learned policy beats never reorganising, always reorganising and picking the single best layout
up front (and strictly beats the frozen layout when the penalty is low enough), the policy is
complete over the state space, learning is reproducible from its seed, degenerate instances
(single column, single query, reorganisation forbidden) reduce to their closed forms, and a
20 layout, 30 query instance is solved and compared against all baselines.
"""
import itertools

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


class Rng:
    def __init__(self, seed):
        self.state = seed & MASK64

    def u64(self):
        self.state, value = splitmix64(self.state)
        return value

    def below(self, bound):
        return self.u64() % bound

    def uniform(self):
        return (self.u64() >> 11) / 9007199254740992.0


class ModelError(Exception):
    """Raised for a malformed storage instance."""


def normalise(layout):
    """A layout is a sorted tuple of blocks, each block a sorted tuple of columns."""
    return tuple(sorted(tuple(sorted(block)) for block in layout))


def all_partitions(items):
    """Every partition of `items` as a layout - the full candidate space for small instances."""
    if not items:
        yield ()
        return
    first, rest = items[0], items[1:]
    for partition in all_partitions(rest):
        yield normalise(((first,),) + partition)
        for index in range(len(partition)):
            merged = list(partition)
            merged[index] = tuple(sorted(merged[index] + (first,)))
            yield normalise(tuple(merged))


class Instance:
    """Columns, a workload cycle and a candidate layout set with an explicit cost model."""

    def __init__(self, widths, queries, layouts, penalty):
        if not widths:
            raise ModelError("an instance needs at least one column")
        if not queries:
            raise ModelError("an instance needs at least one query")
        if penalty < 0:
            raise ModelError("the reorganisation penalty cannot be negative")
        self.widths = list(widths)
        self.columns = tuple(range(len(widths)))
        self.queries = [frozenset(query) for query in queries]
        if any(not query for query in self.queries):
            raise ModelError("a query must read at least one column")
        for query in self.queries:
            if not query <= set(self.columns):
                raise ModelError("a query names a column that does not exist")
        self.penalty = penalty
        self.layouts = [normalise(layout) for layout in layouts]
        if len(set(self.layouts)) != len(self.layouts):
            raise ModelError("the candidate layouts must be distinct")
        for layout in self.layouts:
            if sorted(column for block in layout for column in block) != list(self.columns):
                raise ModelError("every layout must cover every column exactly once")
        self.period = len(self.queries)

    # ------------------------------------------------------------- cost model
    def block_of(self, layout, column):
        for index, block in enumerate(layout):
            if column in block:
                return index
        raise ModelError("column %d is not in the layout" % column)

    def query_cost(self, layout, query):
        touched = {self.block_of(layout, column) for column in query}
        scan = sum(self.widths[column] for column in query)
        return len(touched) + scan

    def reorg_cost(self, before, after):
        """A column counts as moved unless the block it lives in is exactly the same set."""
        moved = 0
        for column in self.columns:
            if before[self.block_of(before, column)] != after[self.block_of(after, column)]:
                moved += 1
        return self.penalty * moved

    def step(self, state, action):
        """Deterministic transition: (position, layout index) plus the chosen layout index."""
        position, index = state
        layout = self.layouts[index]
        target = self.layouts[action]
        query = self.queries[position]
        reward = -(self.query_cost(target, query) + self.reorg_cost(layout, target))
        if position + 1 == self.period:
            return None, reward
        return (position + 1, action), reward

    def state_space(self):
        return [(position, index) for position in range(self.period)
                for index in range(len(self.layouts))]


# ------------------------------------------------------------------- solvers
def dynamic_program(instance):
    """Exact backward induction over the horizon; returns values and the optimal policy."""
    layouts = instance.layouts
    period = instance.period
    values = [[0.0] * len(layouts) for _ in range(period + 1)]
    policy = [[None] * len(layouts) for _ in range(period)]
    for position in range(period - 1, -1, -1):
        query = instance.queries[position]
        for index, layout in enumerate(layouts):
            best_value = None
            best_action = None
            for action, target in enumerate(layouts):
                total = (instance.query_cost(target, query)
                         + instance.reorg_cost(layout, target)
                         + values[position + 1][action])
                if best_value is None or total < best_value:
                    best_value, best_action = total, action
            values[position][index] = best_value
            policy[position][index] = best_action
    flat = {(position, index): policy[position][index]
            for position in range(period) for index in range(len(layouts))}
    return values, flat


def bellman_residual(instance, values):
    """Maximum deviation from the Bellman optimality equation over all states."""
    worst = 0.0
    for position in range(instance.period):
        for index, layout in enumerate(instance.layouts):
            best = None
            for action, target in enumerate(instance.layouts):
                total = (instance.query_cost(target, instance.queries[position])
                         + instance.reorg_cost(layout, target)
                         + values[position + 1][action])
                best = total if best is None else min(best, total)
            worst = max(worst, abs(best - values[position][index]))
    return worst


def exhaustive_optimum(instance):
    """Brute force over every action sequence of one full cycle, starting from layout zero."""
    best = None
    best_sequence = None
    for sequence in itertools.product(range(len(instance.layouts)), repeat=instance.period):
        total = 0
        layout = instance.layouts[0]
        for position, action in enumerate(sequence):
            target = instance.layouts[action]
            total += instance.query_cost(target, instance.queries[position])
            total += instance.reorg_cost(layout, target)
            layout = target
        if best is None or total < best:
            best, best_sequence = total, sequence
    return best, best_sequence


def q_learning(instance, seed, episodes=400, epsilon_start=1.0, epsilon_floor=0.0):
    """Tabular Q-learning with exploring starts and an averaging learning rate."""
    rng = Rng(seed)
    layouts = instance.layouts
    states = instance.state_space()
    q_values = {state: [0.0] * len(layouts) for state in states}
    visits = {state: [0] * len(layouts) for state in states}
    history = []
    for episode in range(episodes):
        epsilon = max(epsilon_floor, epsilon_start * (1.0 - episode / episodes))
        layout = episode % len(layouts)      # deterministic exploring starts
        for position in range(instance.period):
            state = (position, layout)
            if rng.uniform() < epsilon:
                action = rng.below(len(layouts))
            else:
                action = index_of_max(q_values[state])
            reward = -(instance.query_cost(layouts[action], instance.queries[position])
                       + instance.reorg_cost(layouts[layout], layouts[action]))
            visits[state][action] += 1
            alpha = 1.0 / visits[state][action]
            next_state = None if position + 1 == instance.period else (position + 1, action)
            target = reward if next_state is None else reward + max(q_values[next_state])
            q_values[state][action] += alpha * (target - q_values[state][action])
            layout = action
        if episode % max(1, episodes // 10) == 0 or episode == episodes - 1:
            history.append(greedy_cost(instance, q_values, cycles=1))
    return q_values, history


def index_of_max(values):
    best = 0
    for index in range(1, len(values)):
        if values[index] > values[best]:
            best = index
    return best


def greedy_policy_from_q(instance, q_values):
    policy = {}
    for position in range(instance.period):
        for index in range(len(instance.layouts)):
            policy[(position, index)] = index_of_max(q_values[(position, index)])
    return policy


def rollout(instance, policy, cycles=10, start=0):
    """Run a policy for whole cycles and report the total cost and the reorganisation count."""
    index = start
    total = 0
    moves = 0
    for _ in range(cycles):
        for position in range(instance.period):
            action = policy[(position, index)]
            target = instance.layouts[action]
            total += instance.query_cost(target, instance.queries[position])
            total += instance.reorg_cost(instance.layouts[index], target)
            if action != index:
                moves += 1
            index = action
    return total, moves


def greedy_cost(instance, q_values, cycles=3):
    return rollout(instance, greedy_policy_from_q(instance, q_values), cycles)[0]


def value_of_policy(instance, policy, start=0):
    """The cost of one full cycle when the policy is followed from the initial state."""
    return rollout(instance, policy, cycles=1, start=start)[0]


# --------------------------------------------------------------- instances
def build_instance(column_count, query_count, seed, penalty, exhaustive=False, max_columns=5,
                   max_layouts=None):
    rng = Rng(seed)
    widths = [1 + rng.below(6) for _ in range(column_count)]
    queries = []
    for _ in range(query_count):
        size = 1 + rng.below(column_count)
        queries.append(sorted({rng.below(column_count) for _ in range(size * 2)})[:size])
    if exhaustive and column_count <= max_columns:
        layouts = list(all_partitions(tuple(range(column_count))))
    else:
        layouts = []
        layouts.append([tuple(range(column_count))])
        layouts.append([(column,) for column in range(column_count)])
        for query in queries:
            grouped = [tuple(query)] + [(column,) for column in range(column_count)
                                        if column not in query]
            layouts.append(normalise(grouped))
        while len(layouts) < 20:
            shuffled = list(range(column_count))
            for index in range(column_count - 1, 0, -1):
                swap = rng.below(index + 1)
                shuffled[index], shuffled[swap] = shuffled[swap], shuffled[index]
            cut = 1 + rng.below(column_count)
            blocks = []
            cursor = 0
            while cursor < column_count:
                remaining = column_count - cursor
                size = 1 + rng.below(remaining)
                blocks.append(tuple(sorted(shuffled[cursor:cursor + size])))
                cursor += size
            layouts.append(normalise(blocks))
        seen = []
        for layout in layouts:
            candidate = normalise(layout)
            if candidate not in seen:
                seen.append(candidate)
        layouts = seen
        if max_layouts is not None:
            layouts = layouts[:max_layouts]
    return Instance(widths, queries, layouts, penalty)


# ----------------------------------------------------------------------- main
def main():
    # --- 1. the cost model --------------------------------------------------
    instance = Instance([1, 1, 1], [(0, 1), (1, 2), (0, 2)], all_partitions((0, 1, 2)), 2)
    check(len(instance.layouts) == 5, "three columns have five partitions (Bell number)")
    row_store = normalise([(0, 1, 2)])
    column_store = normalise([(0,), (1,), (2,)])
    check(instance.query_cost(row_store, frozenset({0, 1})) == 1 + 2,
          "one block plus the byte widths of the two columns")
    check(instance.query_cost(column_store, frozenset({0, 1})) == 2 + 2,
          "two blocks when the columns are separated")
    check(instance.query_cost(row_store, frozenset({0, 1, 2}))
          < instance.query_cost(column_store, frozenset({0, 1, 2})),
          "clustering co-accessed columns must be cheaper for that query")
    check(instance.reorg_cost(row_store, column_store) == 2 * 3,
          "moving all three columns must cost three times the penalty")
    check(instance.reorg_cost(row_store, row_store) == 0, "keeping the layout must be free")
    check(instance.block_of(row_store, 2) == 0, "a single block holds every column")
    for bad in (lambda: Instance([], [(0,)], [()], 1),
                lambda: Instance([1, 1], [], [[(0,), (1,)]], 1),
                lambda: Instance([1, 1], [(0,)], [[(0,), (1,)]], -1),
                lambda: Instance([1, 1], [(5,)], [[(0,), (1,)]], 1),
                lambda: Instance([1, 1], [()], [[(0,), (1,)]], 1),
                lambda: Instance([1, 1], [(0,)], [[(0,)], [(0,), (1,)]], 1)):
        raised = False
        try:
            bad()
        except ModelError:
            raised = True
        check(raised, "an invalid instance must be rejected")

    # --- 2. transitions and rewards are deterministic -----------------------
    steady = Instance([2, 2], [(0,), (1,)], all_partitions((0, 1)), 1)
    for state in steady.state_space():
        for action in range(len(steady.layouts)):
            first = steady.step(state, action)
            second = steady.step(state, action)
            check(first == second, "the transition must be a function of state and action")
            next_state, reward = first
            check(reward <= 0, "costs are modelled as negative rewards")
            if next_state is not None:
                check(next_state[0] == state[0] + 1, "the position must advance by one query")
                check(next_state[1] == action, "the layout must become the chosen one")
    terminal, terminal_reward = steady.step((1, 0), 0)
    check(terminal is None and terminal_reward <= 0, "the last query must lead to the terminal state")

    # --- 3. the dynamic program is exact ------------------------------------
    values, policy = dynamic_program(instance)
    check(bellman_residual(instance, values) < 1e-9,
          "the dynamic program must satisfy the Bellman optimality equation")
    optimum, sequence = exhaustive_optimum(instance)
    check(values[0][0] == optimum,
          "the dynamic program must match exhaustive enumeration: %r vs %r"
          % (values[0][0], optimum))
    check(all(policy[(position, index)] is not None
              for position in range(instance.period) for index in range(len(instance.layouts))),
          "the policy must be complete over the state space")
    check(all(0 <= action < len(instance.layouts) for action in policy.values()),
          "every policy action must be a valid layout")
    for position in range(instance.period - 1):
        for index in range(len(instance.layouts)):
            check(values[position][index] >= values[position + 1][index] - 10 ** 6,
                  "values must stay finite and ordered")

    # --- 4. reorganisation accounting ---------------------------------------
    check(value_of_policy(instance, policy) == values[0][0],
          "one cycle of the optimal policy must cost exactly the optimal value")
    total, moves = rollout(instance, policy, cycles=4)
    check(0 <= moves <= 4 * instance.period, "the reorganisation count must be bounded")
    lowest = min(values[0])
    highest = max(values[0])
    check(4 * lowest <= total <= 4 * highest,
          "a four cycle roll-out must stay between the best and worst starting layouts")
    frozen = {state: state[1] for state in instance.state_space()}
    frozen_total, frozen_moves = rollout(instance, frozen, cycles=4)
    check(frozen_moves == 0, "the frozen policy must never reorganise")
    check(frozen_total > 0, "the frozen policy must still pay for queries")

    # --- 5. two baselines ---------------------------------------------------
    myopic = {}
    for position in range(instance.period):
        for index, layout in enumerate(instance.layouts):
            best = min(range(len(instance.layouts)),
                       key=lambda action: (instance.query_cost(
                           instance.layouts[action], instance.queries[position])
                           + instance.reorg_cost(layout, instance.layouts[action]), action))
            myopic[(position, index)] = best
    myopic_total, myopic_moves = rollout(instance, myopic, cycles=4)
    check(values[0][0] <= frozen_total / 4,
          "the joint optimum must not lose to the frozen layout")
    check(values[0][0] <= value_of_policy(instance, myopic),
          "the joint optimum must not lose to the myopic policy: %r vs %r"
          % (values[0][0], value_of_policy(instance, myopic)))

    # --- 6. Q-learning rediscovers the optimum ------------------------------
    q_values, history = q_learning(instance, 896001, episodes=400)
    learned = greedy_policy_from_q(instance, q_values)
    learned_total, learned_moves = rollout(instance, learned, cycles=4)
    check(value_of_policy(instance, learned) <= value_of_policy(instance, myopic),
          "learning must not lose to the myopic baseline")
    check(value_of_policy(instance, learned) == values[0][0],
          "learning must find the optimal cycle cost: %d vs %d"
          % (value_of_policy(instance, learned), values[0][0]))
    check(history[-1] == values[0][0],
          "the learning curve must end at the optimum: %r" % (history[-1],))
    check(min(history) >= values[0][0], "no episode may beat the optimum")
    check(history[0] >= history[-1], "the learning curve must not get worse overall")
    check(all(q_values[state][action] <= 0 for state in q_values
              for action in range(len(instance.layouts))),
          "Q values are negated costs, so they can never be positive")

    # --- 7. a harder instance: the joint decision matters -------------------
    hard = build_instance(4, 5, 896100, penalty=1, exhaustive=True)
    hard_values, hard_policy, hard_gap = hard_learning_check(hard, 896200, episodes=4000)
    best_single = min(range(len(hard.layouts)),
                      key=lambda index: sum(hard.query_cost(hard.layouts[index], query)
                                            for query in hard.queries))
    single_policy = {state: best_single for state in hard.state_space()}
    single_total, _ = rollout(hard, single_policy, cycles=3)
    joint_total, joint_moves = rollout(hard, hard_policy, cycles=3)
    check(near_optimal(hard, hard_policy, hard_values),
          "the learned policy must stay within five percent of the exact optimum")
    check(hard_values[0][0] <= value_of_policy(hard, single_policy),
          "the exact joint optimum must not lose to the best single layout")
    expensive = build_instance(4, 5, 896100, penalty=200, exhaustive=True)
    expensive_values, expensive_policy = dynamic_program(expensive)
    check(bellman_residual(expensive, expensive_values) < 1e-9,
          "the expensive instance must satisfy the Bellman equation")
    check(value_of_policy(expensive, expensive_policy) == expensive_values[0][0],
          "the exact optimum must roll out exactly")
    check(expensive_values[0][0] > hard_values[0][0],
          "a high penalty must make the same workload more expensive to serve")

    # --- 8. degenerate instances -------------------------------------------
    single_column = Instance([3], [(0,)], all_partitions((0,)), 5)
    single_values, single_policy = dynamic_program(single_column)
    single_total, single_moves = rollout(single_column, single_policy, cycles=2)
    check(single_moves == 0, "one column cannot be reorganised into anything")
    check(single_total == 2 * (1 + 3), "the closed form for a single column and one query")
    single_query = Instance([1, 1], [(0, 1)], all_partitions((0, 1)), 3)
    single_query_values, _ = dynamic_program(single_query)
    check(single_query_values[0][0] == 2 + 2,
          "with a penalty of three per column it is cheaper to read two blocks than to rewrite")
    check(single_query_values[0][0] < single_query.query_cost(single_query.layouts[1],
                                                             frozenset({0, 1}))
          + single_query.reorg_cost(single_query.layouts[0], single_query.layouts[1]),
          "the clustered layout stays out of reach because the rewrite costs more than it saves")
    forbidden = {state: state[1] for state in single_query.state_space()}
    forbidden_total, forbidden_moves = rollout(single_query, forbidden, cycles=5)
    check(forbidden_moves == 0 and forbidden_total == 5 * (2 + 2),
          "a frozen layout must pay the split-block cost on every cycle")

    # --- 9. scale -----------------------------------------------------------
    scale = build_instance(20, 25, 896300, penalty=2, max_layouts=12)
    check(len(scale.layouts) >= 8,
          "the scaled instance must offer a real layout choice: %d" % len(scale.layouts))
    check(len(scale.state_space()) >= 200,
          "the scaled state space must be substantial: %d" % len(scale.state_space()))
    scale_values, scale_policy = dynamic_program(scale)
    check(bellman_residual(scale, scale_values) < 1e-9,
          "the scaled dynamic program must satisfy the Bellman equation")
    check(value_of_policy(scale, scale_policy) == scale_values[0][0],
          "the scaled policy must roll out to its own value")
    worst_start = max(range(len(scale.layouts)), key=lambda index: scale_values[0][index])
    frozen_worst, frozen_moves = rollout(
        scale, {state: worst_start for state in scale.state_space()},
        start=worst_start, cycles=3)
    check(frozen_moves == 0, "a frozen policy must never reorganise")
    check(3 * scale_values[0][worst_start] < frozen_worst,
          "the joint policy must beat staying on a badly chosen layout: %r vs %d"
          % (3 * scale_values[0][worst_start], frozen_worst))
    best_single = min(range(len(scale.layouts)),
                      key=lambda index: sum(scale.query_cost(scale.layouts[index], query)
                                            for query in scale.queries))
    single_policy = {state: best_single for state in scale.state_space()}
    check(scale_values[0][0] <= value_of_policy(scale, single_policy),
          "the exact joint optimum must not lose to the best single layout")

    # --- 9b. learning on a medium instance ---------------------------------
    medium = build_instance(6, 8, 896450, penalty=2, max_layouts=10)
    medium_values, _ = dynamic_program(medium)
    medium_q, medium_history = q_learning(medium, 896500, episodes=6000)
    medium_learned = greedy_policy_from_q(medium, medium_q)
    medium_gap = value_of_policy(medium, medium_learned) - medium_values[0][0]
    check(0 <= medium_gap <= 0.05 * medium_values[0][0],
          "learning must come within five percent on a medium instance: %d vs %d"
          % (value_of_policy(medium, medium_learned), medium_values[0][0]))
    check(len(medium_history) >= 5, "the medium learning curve must be sampled")
    check(min(medium_history) >= medium_values[0][0],
          "no sampled medium episode may beat the optimum")

    # --- 10. determinism ----------------------------------------------------
    def replay(seed):
        local = build_instance(6, 5, 896500, penalty=2)
        local_values, local_policy = dynamic_program(local)
        local_q, _ = q_learning(local, seed, episodes=200)
        learned_policy = greedy_policy_from_q(local, local_q)
        return (local_values[0][0], rollout(local, local_policy, cycles=2),
                rollout(local, learned_policy, cycles=2),
                [round(local_q[state][0], 9) for state in local.state_space()[:6]])

    check(replay(896600) == replay(896600), "an identical seed must replay the whole solution")

    print("instance: %d columns, %d queries per cycle, %d candidate layouts, penalty %d"
          % (len(instance.columns), instance.period, len(instance.layouts), instance.penalty))
    print("optimum: %r per cycle, exhaustive enumeration %r, policy moves %d"
          % (values[0][0], optimum, moves))
    print("baselines: frozen %d, myopic %d, learned %d over four cycles"
          % (frozen_total, myopic_total, learned_total))
    print("learning curve: %r" % (history,))
    print("scale: %d columns, %d queries, %d layouts, %d states, optimum %r per cycle"
          % (len(scale.columns), scale.period, len(scale.layouts), len(scale.state_space()),
             scale_values[0][0]))
    print("medium instance: learned %d versus optimal %d (gap %d) after 6000 episodes"
          % (value_of_policy(medium, medium_learned), medium_values[0][0], medium_gap))
    print("PASS")


def near_optimal(instance, policy, values, start=0, tolerance=0.05):
    """Is the policy's cycle cost from `start` within the tolerance of the exact optimum?"""
    best = values[0][start]
    cost = value_of_policy(instance, policy, start=start)
    return 0 <= cost - best <= tolerance * best


def hard_learning_check(instance, seed, episodes=900):
    """Solve a larger instance exactly and by learning, returning both for comparison."""
    values, policy = dynamic_program(instance)
    q_values, _ = q_learning(instance, seed, episodes=episodes)
    learned = greedy_policy_from_q(instance, q_values)
    learned_policy_value = value_of_policy(instance, learned)
    gap = learned_policy_value - values[0][0]
    check(0 <= gap <= 0.05 * values[0][0],
          "learning must come within five percent of the harder optimum: %d vs %d"
          % (learned_policy_value, values[0][0]))
    return values, learned, gap


if __name__ == "__main__":
    main()
