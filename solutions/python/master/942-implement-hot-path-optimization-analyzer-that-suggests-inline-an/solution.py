"""Task 942 - Implement hot-path optimization analyzer that suggests inline and caching opportunities.  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)

A profile-driven optimizer.  A profile lists functions with their static size, call count, self cost
per call, number of call sites, distinct-argument count (the cache hit-rate proxy) and their properties
(pure, recursive, virtual).  The analyzer proposes two rewrites:
 * inline - a hot, small function called from at most two sites, never recursive, never virtual; the
   saving is calls * (call overhead - size * i-cache penalty), so a fat body is never inlined;
 * cache  - a hot, pure function whose repeats hit, saving (calls - distinct) * (self cost - lookup).
A function receives at most one rewrite (the larger saving wins, ties go to inline) and the plan is
sorted by saving, then action, then name.

Simplified versus a production advisor (PGO/BOLT): the cost model is a small linear formula instead of
hardware counters and nothing is emitted.  Verified: the plan on a constructed profile is exactly the
expected rewrite set; the modelled savings equal the delta of an independent execution-cost simulation
per suggestion and for the whole plan; recursive, virtual, oversized, cold, many-call-site, impure,
all-distinct-argument and caller-less functions are rejected; the thresholds are inclusive; raising the
hotness threshold only removes suggestions; no function is rewritten twice; the plan order is
deterministic.
"""
CALL_OVERHEAD = 3.0         # cost units removed by inlining one call
INLINE_PENALTY = 0.05       # i-cache penalty per static node per call
SIZE_LIMIT = 40             # nodes; larger bodies are not worth inlining
MAX_CALL_SITES = 2
HOT_CALLS = 1000
CACHE_HIT_MIN = 0.5         # minimum (calls - distinct) / calls to consider memoization
CACHE_LOOKUP = 0.2


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Function:
    __slots__ = ("name", "size", "calls", "self_cost", "sites", "distinct", "recursive", "virtual",
                 "pure")

    def __init__(self, name, size, calls, self_cost, sites=1, distinct=None, recursive=False,
                 virtual=False, pure=True):
        self.name, self.size, self.calls, self.self_cost = name, size, calls, self_cost
        self.sites = sites
        self.distinct = calls if distinct is None else distinct
        self.recursive, self.virtual, self.pure = recursive, virtual, pure

    def hit_rate(self):
        return (self.calls - self.distinct) / self.calls if self.calls else 0.0


def inline_saving(function, hot_calls=HOT_CALLS):
    """Saving from removing the call overhead, net of the i-cache penalty."""
    if function.recursive or function.virtual or function.size > SIZE_LIMIT:
        return 0.0
    if not 1 <= function.sites <= MAX_CALL_SITES or function.calls < hot_calls:
        return 0.0
    return max(0.0, function.calls * (CALL_OVERHEAD - INLINE_PENALTY * function.size))


def cache_saving(function, hot_calls=HOT_CALLS):
    """Saving from memoizing the repeated calls of a pure function."""
    if not function.pure or function.calls < hot_calls or function.hit_rate() < CACHE_HIT_MIN:
        return 0.0
    return max(0.0, (function.calls - function.distinct) * (function.self_cost - CACHE_LOOKUP))


def analyze(functions, hot_calls=HOT_CALLS):
    """Return the plan [(action, name, saving)] with at most one rewrite per function."""
    plan = []
    for function in functions:
        inline = inline_saving(function, hot_calls)
        cache = cache_saving(function, hot_calls)
        if max(inline, cache) <= 0:
            continue
        plan.append(("inline" if inline >= cache else "cache", function.name, max(inline, cache)))
    return sorted(plan, key=lambda entry: (-entry[2], entry[0], entry[1]))


def execution_cost(functions, inlined=(), cached=()):
    """Independent simulation of the profile cost under a rewrite plan."""
    total = 0.0
    for function in functions:
        if function.name in inlined:
            total += function.calls * (function.self_cost + INLINE_PENALTY * function.size)
        elif function.name in cached:
            total += (function.calls * CALL_OVERHEAD + function.distinct * function.self_cost
                      + (function.calls - function.distinct) * CACHE_LOOKUP)
        else:
            total += function.calls * (function.self_cost + CALL_OVERHEAD)
    return total


def baseline_cost(functions):
    return sum(function.calls * (function.self_cost + CALL_OVERHEAD) for function in functions)


def plan_delta(functions, plan):
    """Cost removed by the plan, measured by the simulation rather than by the model."""
    return baseline_cost(functions) - execution_cost(
        functions, {name for action, name, _ in plan if action == "inline"},
        {name for action, name, _ in plan if action == "cache"})


def single_delta(functions, action, name):
    return baseline_cost(functions) - execution_cost(
        functions, {name} if action == "inline" else (), {name} if action == "cache" else ())


def profile():
    return [
        Function("leaf_hash", 12, 50_000, 0.8),
        Function("big_kernel", 60, 20_000, 2.0, distinct=100),
        Function("recursive_walk", 10, 30_000, 1.0, distinct=500, recursive=True),
        Function("hot_impure", 10, 40_000, 1.0, pure=False),
        Function("many_callers", 10, 40_000, 1.0, sites=5),
        Function("cold_util", 10, 100, 1.0, distinct=100),
        Function("virtual_runner", 10, 40_000, 1.0, virtual=True),
        Function("distinct_args_fn", 10, 10_000, 5.0, distinct=9_000),
    ]


def main():
    functions = profile()
    plan = analyze(functions)
    actions = {(action, name) for action, name, _ in plan}

    # 1) happy path: the plan is exactly the expected rewrite set on the constructed profile.
    expected = {("inline", "leaf_hash"), ("inline", "hot_impure"), ("inline", "distinct_args_fn"),
                ("cache", "big_kernel"), ("cache", "recursive_walk")}
    check(actions == expected and len(plan) == len(expected),
          f"unexpected plan: {sorted(actions)}")

    # 2) independent oracle: modelled savings equal the simulated delta, per suggestion and total.
    modelled = sum(entry[2] for entry in plan)
    per_suggestion = sum(single_delta(functions, entry[0], entry[1]) for entry in plan)
    check(abs(per_suggestion - modelled) < 1e-9 and abs(plan_delta(functions, plan) - modelled) < 1e-9
          and modelled > 0,
          f"model {modelled:.1f} vs simulated per-suggestion {per_suggestion:.1f} / "
          f"plan {plan_delta(functions, plan):.1f}")

    # 3) negative checks: every guard really rejects its case.
    rejected = {(action, function.name) for function in functions
                for action, saving in (("inline", inline_saving(function)),
                                       ("cache", cache_saving(function))) if saving <= 0}
    check(("inline", "recursive_walk") in rejected and ("inline", "virtual_runner") in rejected
          and ("inline", "big_kernel") in rejected and ("inline", "cold_util") in rejected
          and ("inline", "many_callers") in rejected
          and ("cache", "hot_impure") in rejected and ("cache", "many_callers") in rejected
          and ("cache", "distinct_args_fn") in rejected and ("cache", "cold_util") in rejected,
          f"guard failures, rejected={sorted(rejected)}")

    # 4) one rewrite per function, positive savings, and the chosen rule is the better one.
    names = [entry[1] for entry in plan]
    chosen = {entry[1]: entry for entry in plan}
    check(len(names) == len(set(names)) and all(entry[2] > 0 for entry in plan)
          and all(chosen[function.name][2] == max(inline_saving(function), cache_saving(function))
                  for function in functions if function.name in chosen),
          f"a function must be rewritten at most once, with its best saving: {plan}")

    # 5) plan ordering: by saving, then action, then name.
    check(plan == sorted(plan, key=lambda entry: (-entry[2], entry[0], entry[1]))
          and [entry[0] for entry in plan] .count("inline") == 3,
          f"plan ordering is wrong: {plan}")

    # 6) boundaries: exactly at the thresholds is hot enough, one step below is not.
    edge = Function("edge", SIZE_LIMIT, HOT_CALLS, 1.0, distinct=HOT_CALLS // 2)
    check(inline_saving(edge) > 0 and cache_saving(edge) > 0
          and inline_saving(Function("cold", 10, HOT_CALLS - 1, 1.0)) == 0.0
          and inline_saving(Function("fat", SIZE_LIMIT + 1, HOT_CALLS, 1.0)) == 0.0
          and cache_saving(Function("cold_cache", 10, HOT_CALLS - 1, 9.0, distinct=1)) == 0.0
          and cache_saving(Function("extreme", 10, HOT_CALLS, 0.1, distinct=1)) == 0.0,
          "the hotness, size and break-even thresholds must be inclusive at the boundary")

    # 7) monotonicity: raising the hotness threshold only removes suggestions.
    previous = {entry[1] for entry in plan}
    for threshold in (2_000, 10_000, 30_000):
        trimmed = {entry[1] for entry in analyze(functions, threshold)}
        check(trimmed <= previous, f"threshold {threshold} must only remove suggestions")
        previous = trimmed

    # 8) degenerate profiles, a free plan benefit, and determinism.
    check(analyze([]) == [] and analyze([Function("root", 10, 50_000, 1.0, sites=0)]) == []
          and analyze(functions) == plan and plan_delta(functions, plan) <= baseline_cost(functions)
          and execution_cost(functions, {"leaf_hash"}, {"big_kernel"}) > 0,
          "the empty profile and a caller-less function must produce no suggestions")

    print(f"plan={[(action, name, round(saving, 1)) for action, name, saving in plan]}")
    print(f"baseline={baseline_cost(functions):.0f} modelled={modelled:.0f} "
          f"simulated={plan_delta(functions, plan):.0f}")
    print("PASS")


if __name__ == "__main__":
    main()
