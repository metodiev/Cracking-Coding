"""Task 969 - Implement curriculum mapping to academic topics and pacing suggestions.  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)

Maps a problem set onto academic topic standards and proposes a paced schedule.  Topics form a
prerequisite DAG; a topic's stage index is the length of the longest prerequisite chain ending at it
(so an academic unit never contains a topic whose prerequisites are taught later).  Problems are mapped
to a topic, its standard code (CS.DS.x) and that stage; problems on topics without a standard are
*reported* as off-curriculum instead of being silently dropped.  Pacing packs each stage's problems
into study weeks of a fixed minute budget without ever mixing two stages in one week, giving an ordered
week plan, per-week loads and a total duration in weeks.

Simplified versus a real LMS scheduler: one global weekly budget, no teacher/room constraints, no
spacing of individual problems across weeks.  Verified: the topological order and stage indices agree
with an independent memoised-recursion oracle, prerequisite cycles are rejected, every problem is mapped
once, the paced plan conserves minutes and never mixes stages, the first-fit-decreasing week count equals
an exact bitmask-DP bin-packing optimum per stage, off-curriculum topics are reported, and empty and
single-problem curricula behave.
"""

STANDARDS = {"arrays": "CS.DS.1", "hashing": "CS.DS.2", "sorting": "CS.DS.3",
             "binary-search": "CS.DS.4", "graphs": "CS.DS.5", "dp": "CS.DS.6"}
# (prerequisite, dependent) - the academic ordering the curriculum must respect
PREREQS = (("arrays", "hashing"), ("arrays", "sorting"), ("sorting", "binary-search"),
           ("hashing", "graphs"), ("binary-search", "dp"), ("graphs", "dp"))
# (problem number, topic, expected minutes for an average learner)
PROBLEMS = ((101, "arrays", 20), (102, "arrays", 15), (103, "arrays", 10), (104, "arrays", 30),
            (105, "hashing", 25), (106, "hashing", 40), (107, "hashing", 15),
            (108, "sorting", 30), (109, "sorting", 20),
            (110, "binary-search", 25), (111, "binary-search", 35),
            (112, "graphs", 45), (113, "graphs", 20),
            (114, "dp", 50), (115, "dp", 40), (116, "dp", 30))
WEEK_BUDGET = 90


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def expect(error, call, message):
    try:
        call()
    except error:
        return
    raise AssertionError(message)


def all_topics(prereqs):
    return sorted({pre for pre, _ in prereqs} | {post for _, post in prereqs})


def topo_order(topics, prereqs):
    """Kahn's algorithm with a sorted ready set, so the order never depends on dict iteration."""
    successors = {topic: [] for topic in topics}
    indegree = {topic: 0 for topic in topics}
    for pre, post in prereqs:
        successors[pre].append(post)
        indegree[post] += 1
    ready = sorted(topic for topic in topics if not indegree[topic])
    order = []
    while ready:
        topic = ready.pop(0)
        order.append(topic)
        for successor in sorted(successors[topic]):
            indegree[successor] -= 1
            if not indegree[successor]:
                ready.append(successor)
        ready.sort()
    if len(order) != len(topics):
        raise ValueError("prerequisite cycle detected: the curriculum cannot be ordered")
    return order


def stage_index(topics, prereqs):
    """Longest prerequisite chain ending at each topic: 0 for a root topic."""
    predecessors = {topic: [] for topic in topics}
    for pre, post in prereqs:
        predecessors[post].append(pre)
    depth = {}
    for topic in topo_order(topics, prereqs):
        chain = predecessors[topic]
        depth[topic] = 0 if not chain else 1 + max(depth[pre] for pre in chain)
    return depth


def stage_index_oracle(topics, prereqs):
    """Independent oracle: plain memoised recursion over the prerequisite relation."""
    predecessors = {topic: [pre for pre, post in prereqs if post == topic] for topic in topics}
    memo = {}

    def visit(topic):
        if topic not in memo:
            chain = predecessors[topic]
            memo[topic] = 1 + max((visit(pre) for pre in chain), default=-1)
        return memo[topic]

    return {topic: visit(topic) for topic in topics}


def map_curriculum(problems, prereqs, standards):
    """problem -> (number, topic, standard, stage 1-based, minutes); off-curriculum topics reported."""
    topics = all_topics(prereqs) + sorted({topic for _, topic, _ in problems if topic not in all_topics(prereqs)})
    depth = stage_index(topics, prereqs)
    mapped, unmapped = [], []
    for number, topic, minutes in sorted(problems):
        if topic not in standards:
            unmapped.append((number, topic))
        else:
            mapped.append((number, topic, standards[topic], depth[topic] + 1, minutes))
    return mapped, unmapped


def pack_stage(items, budget):
    """First-fit-decreasing weeks for one stage: [(numbers, minutes)], never over budget if avoidable."""
    weeks = []
    for number, minutes in sorted(items, key=lambda item: (-item[1], item[0])):
        for week in weeks:
            if week[1] + minutes <= budget:
                week[0].append(number)
                week[1] += minutes
                break
        else:
            weeks.append([[number], minutes])
    return [(tuple(week[0]), week[1]) for week in weeks]


def min_weeks(minutes, budget):
    """Exact oracle: fewest bins of capacity `budget` holding all items (bitmask DP)."""
    count = len(minutes)
    if not count:
        return 0
    bins, loads = [0] * (1 << count), [0] * (1 << count)
    bins[0], loads[0] = 1, 0  # one open bin of capacity `budget` in the empty state
    for mask in range(1, 1 << count):
        best, best_load = 1 << 30, 0
        for index in range(count):
            if mask & (1 << index):
                rest = mask ^ (1 << index)
                used, load = bins[rest], loads[rest]
                if budget - load >= minutes[index]:
                    candidate, candidate_load = used, load + minutes[index]
                else:
                    candidate, candidate_load = used + 1, minutes[index]
                if candidate < best or (candidate == best and candidate_load > best_load):
                    best, best_load = candidate, candidate_load
        bins[mask], loads[mask] = best, best_load
    return bins[(1 << count) - 1]


def build_pacing(mapping, budget):
    """[(stage, problem numbers, minutes)] in stage order - one stage per week, weeks never mixed."""
    stages = {}
    for number, _topic, _standard, stage, minutes in mapping:
        stages.setdefault(stage, []).append((number, minutes))
    plan = []
    for stage in sorted(stages):
        for numbers, minutes in pack_stage(stages[stage], budget):
            plan.append((stage, numbers, minutes))
    return plan


def per_stage_minutes(mapping):
    stages = {}
    for number, _topic, _standard, stage, minutes in mapping:
        stages.setdefault(stage, []).append(minutes)
    return stages


def main():
    prereqs = list(PREREQS)
    topics = all_topics(prereqs)
    order = topo_order(topics, prereqs)
    position = {topic: index for index, topic in enumerate(order)}
    # 1) topological order: a permutation of the topics where every prerequisite comes first.
    check(sorted(order) == topics and all(position[pre] < position[post] for pre, post in prereqs)
          and order == topo_order(topics, prereqs), f"topological order is wrong: {order}")
    # 2) stage indices (longest prerequisite chain) match the memoised-recursion oracle and the hand count.
    depth = stage_index(topics, prereqs)
    check(depth == stage_index_oracle(topics, prereqs)
          and depth == {"arrays": 0, "hashing": 1, "sorting": 1, "binary-search": 2, "graphs": 2, "dp": 3},
          f"stage indices are wrong: {depth}")
    # 3) negative + boundary: a prerequisite cycle is rejected, and trivial curricula stay trivial.
    expect(ValueError, lambda: topo_order(topics, prereqs + [("dp", "arrays")]), "a cycle must be rejected")
    expect(ValueError, lambda: stage_index(["a"], [("a", "a")]), "a self-loop must be rejected")
    check(topo_order([], []) == [] and stage_index([], []) == {} and topo_order(["arrays"], []) == ["arrays"],
          "the empty and single-topic boundaries must be handled")
    # 4) mapping: each problem carries its topic's standard and stage; off-curriculum topics are reported.
    mapped, unmapped = map_curriculum(PROBLEMS, prereqs, STANDARDS)
    mapped_off, unmapped_off = map_curriculum(PROBLEMS + ((117, "strings", 20),), prereqs, STANDARDS)
    check(len(mapped) == len(PROBLEMS) and unmapped == [] and unmapped_off == [(117, "strings")]
          and mapped_off == mapped and sorted(entry[0] for entry in mapped) == sorted(n for n, _, _ in PROBLEMS)
          and all(entry[2] == STANDARDS[entry[1]] and entry[3] == depth[entry[1]] + 1 for entry in mapped),
          f"curriculum mapping is wrong: unmapped={unmapped_off}")
    # 5) pacing: every problem scheduled exactly once, stages never mixed, minute total conserved.
    plan = build_pacing(mapped, WEEK_BUDGET)
    stage_of = {entry[0]: entry[3] for entry in mapped}
    scheduled = [number for _, numbers, _ in plan for number in numbers]
    total_minutes = sum(entry[4] for entry in mapped)
    check(sorted(scheduled) == sorted(stage_of) and len(scheduled) == len(set(scheduled))
          and sum(minutes for _, _, minutes in plan) == total_minutes
          and [stage for stage, _, _ in plan] == sorted(stage for stage, _, _ in plan)
          and all(len({stage_of[number] for number in numbers}) == 1 for _, numbers, _ in plan)
          and all(minutes <= WEEK_BUDGET or len(numbers) == 1 for _, numbers, minutes in plan),
          f"the paced plan is infeasible: {plan}")
    # 6) independent oracle: first-fit-decreasing weeks equal the exact bin-packing optimum per stage.
    optimum = sum(min_weeks(minutes, WEEK_BUDGET) for minutes in per_stage_minutes(mapped).values())
    for stage, minutes in sorted(per_stage_minutes(mapped).items()):
        weeks = [row for row in plan if row[0] == stage]
        check(len(weeks) == min_weeks(minutes, WEEK_BUDGET),
              f"stage {stage} used {len(weeks)} weeks, optimum is {min_weeks(minutes, WEEK_BUDGET)}")
    check(len(plan) == optimum == 7 and sum(minutes for _, _, minutes in plan) == 450,
          f"expected 7 weeks totalling 450 minutes, got {len(plan)} weeks")
    # 7) standards coverage: every curriculum topic has one unique standard and every stage is used.
    check(set(depth) == set(STANDARDS) and len(set(STANDARDS.values())) == len(STANDARDS)
          and {stage for _, _, _, stage, _ in mapped} == set(range(1, max(depth.values()) + 2))
          and all(entry[2] in set(STANDARDS.values()) for entry in mapped),
          "every curriculum topic needs exactly one standard and every stage must be populated")
    # 8) empty curriculum: no mapping, no plan, no oracle weeks; a single problem is one week.
    empty_mapped, empty_unmapped = map_curriculum((), (), STANDARDS)
    check(empty_mapped == [] and empty_unmapped == [] and build_pacing([], WEEK_BUDGET) == []
          and min_weeks([], WEEK_BUDGET) == 0 and pack_stage([], WEEK_BUDGET) == []
          and build_pacing([(1, "arrays", "CS.DS.1", 1, 20)], WEEK_BUDGET) == [(1, (1,), 20)]
          and min_weeks([20], WEEK_BUDGET) == 1,
          "the empty and single-problem curricula must not produce phantom weeks")
    # 9) determinism: re-running the pipeline reproduces the mapping, the order and the week plan.
    check(map_curriculum(PROBLEMS, prereqs, STANDARDS) == (mapped, unmapped)
          and build_pacing(mapped, WEEK_BUDGET) == plan and topo_order(topics, prereqs) == order
          and stage_index_oracle(topics, prereqs) == depth,
          "the pipeline must be reproducible")
    summary = ", ".join(f"stage{stage}:{len([r for r in plan if r[0] == stage])}w"
                        for stage in sorted(per_stage_minutes(mapped)))
    print(f"topics={len(topics)} problems={len(mapped)} off_curriculum={len(unmapped_off)} "
          f"minutes={total_minutes} weeks={len(plan)} ({summary})")
    print(f"order={order} stages={{{', '.join(f'{t}:{d}' for t, d in sorted(depth.items()))}}}")
    print("PASS")


if __name__ == "__main__":
    main()
