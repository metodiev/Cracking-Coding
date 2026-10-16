"""Task 673 - Implement SSA construction and basic optimizations.  (Tier: Expert, Beyond — Compilers, Languages & Solvers)

Minimal SSA construction (Cytron et al.) plus SSA-level cleanups, all verified.

Pipeline
  1. dominator sets by the iterative intersection algorithm, immediate dominators, and
     dominance frontiers (the standard walk over join blocks);
  2. phi placement by the def-site worklist over the dominance frontier;
  3. renaming by a pre-order walk of the dominator tree with one stack of current versions
     per variable: every definition and every phi becomes a fresh version, every use is
     rewritten, and each successor's phi receives one operand per predecessor edge.

Verified invariants: every version is defined exactly once; every use is dominated by the
definition of the version it reads and every phi operand (v on the edge p -> s) is
dominated by v's definition - dominance being re-checked against an independent brute-force
oracle ("delete d, is the node still reachable?"); every phi sits in the dominance frontier
of a block that defines or re-defines its variable; and a deterministic interpreter yields
identical `use` traces for the source program and its SSA form on concrete paths, which is
what makes the construction sound.

Optimizations: trivial-phi elimination (all operands are the same version), dead version
elimination (unread defs and phis, iterated to a fixpoint), and sparse constant propagation
(literal defs plus phis whose operands all agree). Both transformations are re-checked
semantically and against the SSA invariants.

Simplified: scalars only, no arrays or aliasing, a definition carries an optional literal
instead of an expression tree, and the CFG must be fully reachable from the entry block.
"""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def normalize(block):
    out = []
    for ins in block:
        if ins[0] == "def":
            out.append(("def", ins[1], ins[2] if len(ins) > 2 else None))
        elif ins[0] == "use":
            out.append(("use", ins[1], None))
        else:
            raise ValueError("unknown instruction %r" % (ins,))
    return out


class Cfg:
    """Blocks of ("def", var[, literal]) / ("use", var) instructions plus successor lists."""

    def __init__(self, blocks, succ):
        self.blocks = [normalize(block) for block in blocks]
        self.succ = [list(s) for s in succ]
        if len(self.succ) != len(self.blocks):
            raise ValueError("one successor list per block is required")
        for b, targets in enumerate(self.succ):
            if len(set(targets)) != len(targets):
                raise ValueError("duplicate successors in block %d" % b)
            for t in targets:
                if not 0 <= t < len(self.blocks):
                    raise ValueError("edge %d -> %d is out of range" % (b, t))
                if t == b:
                    raise ValueError("self loops must be modelled with a real block")
        self.pred = [[] for _ in self.blocks]
        for b, targets in enumerate(self.succ):
            for t in targets:
                self.pred[t].append(b)

    def variables(self):
        return sorted({var for block in self.blocks for _, var, _ in block})

    def def_sites(self):
        sites = {}
        for b, block in enumerate(self.blocks):
            for op, var, _ in block:
                if op == "def":
                    sites.setdefault(var, []).append(b)
        return sites

    def instruction_count(self):
        return sum(len(block) for block in self.blocks)


class Ssa:
    """Program in SSA form; instructions are phi / def / use over (var, index) versions."""

    def __init__(self, cfg, blocks):
        self.cfg = cfg
        self.blocks = blocks

    def instruction_count(self):
        return sum(len(block) for block in self.blocks)

    def phi_count(self):
        return sum(1 for block in self.blocks for ins in block if ins[0] == "phi")

    def defined_versions(self):
        return [(ins[1], b) for b, block in enumerate(self.blocks) for ins in block
                if ins[0] in ("phi", "def")]

    def def_sites(self):
        return {version: b for version, b in self.defined_versions()}

    def used_versions(self):
        used = []
        for b, block in enumerate(self.blocks):
            for ins in block:
                if ins[0] == "use" and ins[3] is not None:
                    used.append((ins[3], b))
                elif ins[0] == "phi":
                    used.extend((v, p) for p, v in ins[3].items() if v is not None)
        return used


def reachable_from(cfg, start=0):
    seen, stack = {start}, [start]
    while stack:
        for s in cfg.succ[stack.pop()]:
            if s not in seen:
                seen.add(s)
                stack.append(s)
    return seen


def dominators(cfg):
    """Iterative: dom[b] = {b} union the intersection over the predecessors' dom sets."""
    n = len(cfg.blocks)
    dom = [set(range(n)) for _ in range(n)]
    dom[0] = {0}
    changed = True
    while changed:
        changed = False
        for b in range(1, n):
            if not cfg.pred[b]:
                continue
            new = set(range(n))
            for p in cfg.pred[b]:
                new &= dom[p]
            new.add(b)
            if new != dom[b]:
                dom[b], changed = new, True
    return dom


def brute_force_dominators(cfg):
    """Oracle: d dominates x iff deleting d makes x unreachable from the entry."""
    n = len(cfg.blocks)
    dom = [{b} for b in range(n)]
    reachable = reachable_from(cfg)
    for x in reachable - {0}:
        dom[x].add(0)
    for d in range(1, n):
        seen, stack = {0}, [0]
        while stack:
            for s in cfg.succ[stack.pop()]:
                if s != d and s not in seen:
                    seen.add(s)
                    stack.append(s)
        for x in reachable - {0} - seen:
            dom[x].add(d)
    return dom


def immediate_dominators(dom):
    idom = [None] * len(dom)
    for b in range(1, len(dom)):
        strict = dom[b] - {b}
        if strict:
            idom[b] = max(strict, key=lambda d: (len(dom[d]), -d))
    return idom


def dominance_frontiers(cfg, idom):
    df = [set() for _ in cfg.blocks]
    for b in range(len(cfg.blocks)):
        if len(cfg.pred[b]) < 2:
            continue
        for p in cfg.pred[b]:
            runner = p
            while runner is not None and runner != idom[b]:
                df[runner].add(b)
                runner = idom[runner]
    return df


def place_phis(cfg, df, variables):
    """Def-site worklist over the dominance frontier (minimal SSA)."""
    sites = cfg.def_sites()
    phi = {v: set() for v in variables}
    for v in variables:
        work = list(sites.get(v, ()))
        while work:
            for y in sorted(df[work.pop()]):
                if y not in phi[v]:
                    phi[v].add(y)
                    if y not in sites.get(v, ()):
                        work.append(y)
    return phi


def build_ssa(cfg):
    """Returns (ssa, phi placement). Renaming is a pre-order walk of the dominator tree."""
    if len(reachable_from(cfg)) != len(cfg.blocks):
        raise ValueError("every block must be reachable from the entry for SSA renaming")
    variables = cfg.variables()
    idom = immediate_dominators(dominators(cfg))
    phi = place_phis(cfg, dominance_frontiers(cfg, idom), variables)
    n = len(cfg.blocks)
    children = [[] for _ in range(n)]
    for b in range(1, n):
        children[idom[b]].append(b)
    # one operand slot per predecessor edge, filled when the predecessor is renamed
    slots = [[(v, {p: None for p in cfg.pred[b]}) for v in variables if b in phi[v]]
             for b in range(n)]
    counter = {v: 0 for v in variables}
    stacks = {v: [] for v in variables}
    new_blocks = [None] * n
    pushed = [[] for _ in range(n)]
    stack = [(0, "enter")]
    while stack:
        b, action = stack.pop()
        if action == "exit":
            for v, _ in pushed[b]:
                stacks[v].pop()
            continue
        body, pushed_b = [], []
        for v, operands in slots[b]:
            counter[v] += 1
            version = (v, counter[v])
            body.append(("phi", version, v, operands))
            stacks[v].append(version)
            pushed_b.append((v, version))
        for op, var, literal in cfg.blocks[b]:
            if op == "def":
                counter[var] += 1
                version = (var, counter[var])
                body.append(("def", version, var, literal))
                stacks[var].append(version)
                pushed_b.append((var, version))
            else:
                body.append(("use", None, var, stacks[var][-1] if stacks[var] else None))
        new_blocks[b], pushed[b] = body, pushed_b
        for s in cfg.succ[b]:
            for v, operands in slots[s]:
                operands[b] = stacks[v][-1] if stacks[v] else None
        stack.append((b, "exit"))
        for child in reversed(sorted(children[b])):
            stack.append((child, "enter"))
    return Ssa(cfg, new_blocks), phi


def ssa_problems(ssa):
    """Every way the SSA form could be malformed; an empty list means well formed."""
    problems = []
    if len(ssa.def_sites()) != len(ssa.defined_versions()):
        problems.append("a version is defined more than once")
    sites = ssa.def_sites()
    oracle = brute_force_dominators(ssa.cfg)
    for version, b in ssa.used_versions():
        if version not in sites:
            problems.append("use of an undefined version %r" % (version,))
        elif sites[version] not in oracle[b]:
            problems.append("definition of %r does not dominate its use in block %d"
                            % (version, b))
    df = dominance_frontiers(ssa.cfg, immediate_dominators(dominators(ssa.cfg)))
    source = ssa.cfg.def_sites()
    for b, block in enumerate(ssa.blocks):
        for ins in block:
            if ins[0] != "phi":
                continue
            if len(ins[3]) != len(ssa.cfg.pred[b]):
                problems.append("phi in block %d has an incomplete operand list" % b)
            if not any(b in df[x] for x in range(len(ssa.blocks))
                       if x in source.get(ins[2], ()) or
                       any(i[0] == "phi" and i[2] == ins[2] for i in ssa.blocks[x])):
                problems.append("phi for %s in block %d is outside the frontier" % (ins[2], b))
    return problems


def constant_versions(ssa):
    """Sparse constant propagation: literal defs, then phis whose operands all agree."""
    const = {ins[1]: ins[3] for block in ssa.blocks for ins in block
             if ins[0] == "def" and ins[3] is not None}
    changed = True
    while changed:
        changed = False
        for block in ssa.blocks:
            for ins in block:
                if ins[0] != "phi" or ins[1] in const or not ins[3]:
                    continue
                values = {const.get(v) for v in ins[3].values()}
                if len(values) == 1 and None not in values:
                    const[ins[1]] = values.pop()
                    changed = True
    return const


def optimize_ssa(ssa, max_rounds=32):
    """Trivial-phi elimination followed by dead version elimination, to a fixpoint."""
    current = ssa
    stats = {"phis": 0, "defs": 0, "rounds": 0}
    for _ in range(max_rounds):
        stats["rounds"] += 1
        subst = {}

        def resolve(version):
            while version in subst:
                version = subst[version]
            return version

        changed = False
        blocks = []
        for block in current.blocks:
            body = []
            for ins in block:
                if ins[0] == "phi":
                    operands = {p: (resolve(v) if v is not None else None)
                                for p, v in ins[3].items()}
                    distinct = {v for v in operands.values() if v is not None}
                    if len(distinct) == 1 and all(v is not None for v in operands.values()):
                        subst[ins[1]] = distinct.pop()
                        stats["phis"] += 1
                        changed = True
                        continue
                    body.append(("phi", ins[1], ins[2], operands))
                elif ins[0] == "def":
                    body.append(ins)
                else:
                    body.append(("use", None, ins[2], resolve(ins[3]) if ins[3] else None))
            blocks.append(body)
        current = Ssa(current.cfg, blocks)
        used = {version for version, _ in current.used_versions()}
        kept = []
        for block in current.blocks:
            body = []
            for ins in block:
                if ins[0] in ("def", "phi") and ins[1] not in used:
                    stats["phis" if ins[0] == "phi" else "defs"] += 1
                    changed = True
                    continue
                body.append(ins)
            kept.append(body)
        current = Ssa(current.cfg, kept)
        if not changed:
            return current, stats
    raise AssertionError("SSA optimization failed to converge")


def value_of(var, b):
    """Value produced by a definition - independent of the position inside the block."""
    return (ord(var) - ord("a") + 1) * 1000 + b


def run_source(cfg, path):
    env, trace = {}, []
    for b in path:
        for op, var, literal in cfg.blocks[b]:
            if op == "def":
                env[var] = literal if literal is not None else value_of(var, b)
            else:
                trace.append(env.get(var, 0))
    return trace


def run_ssa(ssa, path):
    env, trace = {}, []
    for position, b in enumerate(path):
        previous = path[position - 1] if position else None
        for ins in ssa.blocks[b]:
            if ins[0] == "phi":
                source = ins[3].get(previous) if previous is not None else None
                env[ins[1]] = env.get(source, 0)
            elif ins[0] == "def":
                env[ins[1]] = ins[3] if ins[3] is not None else value_of(ins[2], b)
            else:
                trace.append(env.get(ins[3], 0))
    return trace


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return state, z ^ (z >> 31)


def random_cfg(seed, block_count=5, variables="abc"):
    """Forward DAG plus one back edge; all blocks stay reachable from the entry."""
    state = seed
    succ = [[] for _ in range(block_count)]
    for i in range(block_count - 1):
        succ[i].append(i + 1)
        state, raw = splitmix64(state)
        if raw % 2 == 0 and i + 2 < block_count:
            succ[i].append(i + 2)
    blocks = []
    for _ in range(block_count):
        state, raw = splitmix64(state)
        instructions = []
        for _ in range(1 + raw % 3):
            state, raw = splitmix64(state)
            var = variables[raw % len(variables)]
            if raw % 3 == 0:
                instructions.append(("def", var, raw % 7))
            elif raw % 3 == 1:
                instructions.append(("def", var))
            else:
                instructions.append(("use", var))
        blocks.append(instructions)
    if block_count > 1:
        state, raw = splitmix64(state)
        succ[block_count - 1].append(raw % (block_count - 1))
    return Cfg(blocks, succ)


def random_paths(cfg, count, seed, max_length=18):
    state = seed
    paths = []
    for _ in range(count):
        path = [0]
        for _ in range(max_length):
            targets = cfg.succ[path[-1]]
            if not targets:
                break
            state, raw = splitmix64(state)
            path.append(targets[raw % len(targets)])
        paths.append(path)
    return paths


def main():
    # ---- a diamond: the merge needs exactly one phi ----------------------------------
    diamond = Cfg([[("def", "x", 1)], [("def", "x", 2)], [("use", "x")]], [[1, 2], [2], []])
    ssa, phi = build_ssa(diamond)
    check(phi["x"] == {2} and ssa.phi_count() == 1,
          "the diamond merge must receive exactly one phi: %r" % (phi,))
    check(ssa.blocks[2][0][0] == "phi" and len(ssa.blocks[2][0][3]) == 2,
          "the phi lists one operand per predecessor (0 and 1): %r" % (ssa.blocks[2][0],))
    check(ssa_problems(ssa) == [], "the diamond SSA form must satisfy every invariant")
    check(run_source(diamond, [0, 1, 2]) == run_ssa(ssa, [0, 1, 2]) == [2],
          "the path through block 1 must observe branch 1's definition")
    check(run_source(diamond, [0, 2]) == run_ssa(ssa, [0, 2]) == [1],
          "the direct path must observe the entry's definition")

    # ---- a loop: the header receives a phi -------------------------------------------
    loop = Cfg([[("def", "i", 0)], [("use", "i")], [("def", "i")], [("use", "i")]],
               [[1], [2, 3], [1], []])
    loop_ssa, loop_phi = build_ssa(loop)
    check(loop_phi["i"] == {1} and loop_ssa.blocks[1][0][0] == "phi",
          "the loop header must start with a phi: %r" % (loop_phi,))
    check(ssa_problems(loop_ssa) == [], "loop SSA invariants")
    for path in ([0, 1, 3], [0, 1, 2, 1, 3], [0, 1, 2, 1, 2, 1, 3]):
        check(run_source(loop, path) == run_ssa(loop_ssa, path),
              "the loop phi must be selected per edge on %r" % (path,))
    check(run_ssa(loop_ssa, [0, 1, 2, 1, 3]) == [0, value_of("i", 2), value_of("i", 2)],
          "the second iteration must read the body's definition")

    # ---- random CFGs: dominators vs oracle, SSA invariants, trace preservation --------
    phis_total = 0
    for i in range(120):
        cfg = random_cfg(0xA5A5 + i * 104729, 3 + i % 4)
        dom = dominators(cfg)
        check(dom == brute_force_dominators(cfg), "dominator mismatch on random CFG %d" % i)
        check(all(b in dom[b] for b in range(len(cfg.blocks))), "dominance is reflexive")
        built, _ = build_ssa(cfg)
        problems = ssa_problems(built)
        check(problems == [], "SSA invariant broken on CFG %d: %r" % (i, problems))
        check(len(built.defined_versions()) == len(set(built.defined_versions())),
              "versions must be defined exactly once (CFG %d)" % i)
        phis_total += built.phi_count()
        optimized, _ = optimize_ssa(built)
        check(ssa_problems(optimized) == [], "optimization broke SSA on CFG %d" % i)
        check(optimized.instruction_count() <= built.instruction_count(),
              "optimization must not grow the program (CFG %d)" % i)
        for path in random_paths(cfg, 3, 0x321 + i):
            check(run_source(cfg, path) == run_ssa(built, path) == run_ssa(optimized, path),
                  "SSA or the optimizer changed the trace of CFG %d on %r" % (i, path))
    check(phis_total > 10, "the sweep must really place phis (%d)" % phis_total)

    # ---- a genuinely trivial phi ------------------------------------------------------
    # Minimal SSA rarely produces one, but any pass that splits edges does, so the
    # optimizer is exercised on a hand-built phi whose only operand is its own source.
    hand = Ssa(Cfg([[("def", "x", 4)], [("use", "x")]], [[1], []]),
               [[("def", ("x", 1), "x", 4)],
                [("phi", ("x", 2), "x", {0: ("x", 1)}), ("use", None, "x", ("x", 2))]])
    cleaned, clean_stats = optimize_ssa(hand)
    check(clean_stats["phis"] >= 1 and cleaned.phi_count() == 0,
          "the trivial phi must be eliminated: %r" % (clean_stats,))
    check(run_ssa(cleaned, [0, 1]) == run_ssa(hand, [0, 1]) == [4],
          "eliminating a trivial phi must not change the trace")

    # ---- dead definitions and constant propagation ------------------------------------
    dead_ssa, _ = build_ssa(Cfg([[("def", "x"), ("def", "z")], [("use", "x")]], [[1], []]))
    alive, dead_stats = optimize_ssa(dead_ssa)
    check(alive.instruction_count() == 2 and dead_stats["defs"] == 1,
          "the unused definition of z must be removed: %r" % (dead_stats,))
    check(all(ins[0] != "def" or ins[2] != "z" for block in alive.blocks for ins in block),
          "no definition of z may survive")

    agreeing, _ = build_ssa(Cfg([[("def", "k", 7)], [("def", "k", 7)], [("use", "k")]],
                                [[1, 2], [2], []]))
    check(constant_versions(agreeing).get(agreeing.blocks[2][0][1]) == 7,
          "a phi fed by two equal literals is constant")
    disagreeing, _ = build_ssa(Cfg([[("def", "k", 7)], [("def", "k", 9)], [("use", "k")]],
                                   [[1, 2], [2], []]))
    check(constant_versions(disagreeing).get(disagreeing.blocks[2][0][1]) is None,
          "disagreeing literals must not be folded to a constant")

    # ---- boundary cases, error handling, determinism ----------------------------------
    single_ssa, single_phi = build_ssa(Cfg([[("use", "v")]], [[]]))
    check(single_phi == {"v": set()} and single_ssa.phi_count() == 0,
          "a straight-line program needs no phis")
    check(run_ssa(single_ssa, [0]) == [0], "an undefined read yields the default value")
    no_defs, _ = build_ssa(Cfg([[("use", "v")], [("use", "v")]], [[1], []]))
    check(no_defs.blocks[0][0][3] is None and no_defs.used_versions() == [],
          "reads of never-defined variables stay undefined and are not counted as uses")
    rejected = 0
    for bad in (lambda: Cfg([[("def", "a")]], [[0]]),
                lambda: Cfg([[("def", "a")]], [[1]]),
                lambda: Cfg([[("def", "a")]], []),
                lambda: Cfg([[("frob", "a")]], [[]]),
                lambda: build_ssa(Cfg([[("def", "a")]], [[1, 1]])),
                lambda: build_ssa(Cfg([[("def", "a")], [("def", "b")]], [[], []]))):
        try:
            bad()
        except ValueError:
            rejected += 1
    check(rejected == 6, "malformed programs must be rejected (%d of 6)" % rejected)
    check(build_ssa(random_cfg(0x1234))[0].blocks == build_ssa(random_cfg(0x1234))[0].blocks,
          "SSA construction must be deterministic")

    print("random CFGs: 120/120 constructed, %d phis placed, invariants and traces intact"
          % phis_total)
    print("loop header phi = %r; optimizer removed %d phi(s) and %d dead def(s) in the "
          "hand-built and z cases" % (loop_ssa.blocks[1][0][1], clean_stats["phis"],
                                      dead_stats["defs"]))
    print("PASS")


if __name__ == "__main__":
    main()
