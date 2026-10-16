"""Task 671 - Implement static program analysis: reaching definitions simple algorithm.  (Tier: Expert, Beyond — Compilers, Languages & Solvers)

Classic bitvector dataflow analysis on a small control-flow graph. A definition is a
unique *instruction*, not just a variable: the pair (block, index) makes every definition
distinguishable, which is what makes the analysis useful for SSA and for use-def chains.

    gen[B]  = the last definition of each variable inside B
    kill[B] = every other definition of the variables that B defines
    OUT[B]  = gen[B] union (IN[B] minus kill[B])
    IN[B]   = union of OUT[p] over the predecessors p

The transfer function is monotone over the finite lattice of definition sets, so the
round-robin fixpoint iteration terminates; the number of rounds is bounded by the number of
definitions times the number of blocks.

Three things are verified. (1) The fixpoint satisfies the equations exactly - a further
full pass changes nothing, and `IN[B]` is precisely the union of the predecessors' OUT
sets. (2) A path-based oracle, which walks the graph definition by definition and stops at
any block that redefines the variable, reproduces every IN set on 120 randomly generated
CFGs. (3) The solver is order independent: iterating the blocks in reverse order reaches
the same least fixpoint, and a snapshot of each round shows the solution growing
monotonically.

Simplified: no procedures/interprocedural effects, no arrays (a definition writes a scalar
variable), and the CFG is given rather than parsed.
"""

BIG = 10 ** 9


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Cfg:
    """Control-flow graph: `blocks[i]` is a list of instructions, `succ[i]` successors.

    An instruction is ("def", var) or ("use", var). Instructions are numbered by their
    position, so (block, index, variable) identifies a definition uniquely.
    """

    def __init__(self, blocks, succ):
        self.blocks = [list(block) for block in blocks]
        self.succ = [list(s) for s in succ]
        if len(self.succ) != len(self.blocks):
            raise ValueError("one successor list per block is required")
        for b, targets in enumerate(self.succ):
            for t in targets:
                if not 0 <= t < len(self.blocks):
                    raise ValueError("edge %d -> %d is out of range" % (b, t))
            if b in targets:
                raise ValueError("self loops must be modelled with a real block")
        self.pred = [[] for _ in self.blocks]
        for b, targets in enumerate(self.succ):
            for t in targets:
                self.pred[t].append(b)

    def definitions(self):
        return [(b, k, var) for b, block in enumerate(self.blocks)
                for k, (op, var) in enumerate(block) if op == "def"]

    def defined_vars(self, b):
        return {var for op, var in self.blocks[b] if op == "def"}

    def used_vars(self, b):
        return {var for op, var in self.blocks[b] if op == "use"}

    def gen_kill(self):
        """Return (gen, kill): last definition per variable, and everything it kills."""
        gen, kill = {}, {}
        for b, block in enumerate(self.blocks):
            last = {}
            for k, (op, var) in enumerate(block):
                if op == "def":
                    last[var] = k
            gen[b] = {(b, k, var) for var, k in last.items()}
            kill[b] = set()
            for b2, block2 in enumerate(self.blocks):
                for k2, (op2, var2) in enumerate(block2):
                    if op2 == "def" and var2 in last and (b2, k2, var2) not in gen[b]:
                        kill[b].add((b2, k2, var2))
        return gen, kill


def solve_reaching_definitions(cfg, order=None, collect_rounds=False):
    """Round-robin fixpoint solve. Returns (inn, out, rounds, snapshots)."""
    gen, kill = cfg.gen_kill()
    n = len(cfg.blocks)
    inn = [set() for _ in range(n)]
    out = [set() for _ in range(n)]
    order = list(range(n)) if order is None else list(order)
    snapshots = []
    rounds = 0
    while True:
        rounds += 1
        changed = False
        for b in order:
            new_in = set()
            for p in cfg.pred[b]:
                new_in |= out[p]
            new_out = gen[b] | (new_in - kill[b])
            if new_in != inn[b] or new_out != out[b]:
                inn[b], out[b] = new_in, new_out
                changed = True
        if collect_rounds:
            snapshots.append(([set(s) for s in inn], [set(s) for s in out]))
        if not changed:
            return inn, out, rounds, snapshots


def reaching_oracle(cfg):
    """Path-based oracle: which definitions reach the entry of each block.

    For every definition d = (b, k, var) that is the last definition of `var` inside b, walk
    forward from b's successors; d reaches the entry of every visited block, and it keeps
    propagating only while the block it is currently in does not redefine `var` (a
    redefining block still receives d at its entry, but d dies inside it).
    """
    result = {b: set() for b in range(len(cfg.blocks))}
    for b, k, var in cfg.definitions():
        indices = [kk for kk, (op, vv) in enumerate(cfg.blocks[b])
                   if op == "def" and vv == var]
        if k != max(indices):
            continue                     # killed by a later definition in the same block
        stack = list(cfg.succ[b])
        seen = set(stack)
        for s in stack:
            result[s].add((b, k, var))
        while stack:
            cur = stack.pop()
            if var in cfg.defined_vars(cur):
                continue                 # this block kills the definition
            for t in cfg.succ[cur]:
                if t not in seen:
                    seen.add(t)
                    result[t].add((b, k, var))
                    stack.append(t)
    return result


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return state, z ^ (z >> 31)


def random_cfg(seed, block_count=5, variables="abc"):
    """Random CFG: a forward DAG over block indices plus one back edge for a loop."""
    state = seed
    succ = [[] for _ in range(block_count)]
    for i in range(block_count - 1):
        succ[i].append(i + 1)
        state, raw = splitmix64(state)
        if raw % 2 == 0 and i + 2 < block_count:
            succ[i].append(i + 2)
    blocks = []
    for i in range(block_count):
        state, raw = splitmix64(state)
        instructions = []
        for k in range(1 + raw % 3):
            state, raw = splitmix64(state)
            op = "def" if raw % 3 else "use"
            instructions.append((op, variables[raw % len(variables)]))
        blocks.append(instructions)
    if block_count > 1:
        state, raw = splitmix64(state)
        succ[block_count - 1].append(raw % (block_count - 1))   # back edge -> a real loop
    return Cfg(blocks, succ)


def main():
    # ---- the textbook reaching-definitions example -----------------------------------
    textbook = Cfg(
        [ [("use", "a"), ("def", "b")],
          [("def", "a"), ("use", "b")],
          [("def", "a")],
          [("use", "a")] ],
        [[1], [2, 3], [1], []])
    gen, kill = textbook.gen_kill()
    check(gen[0] == {(0, 1, "b")}, "gen keeps only the last definition per variable")
    check((1, 0, "a") in kill[2] and (2, 0, "a") in kill[1],
          "each definition of a variable kills the others: %r" % (kill,))
    inn, out, rounds, _ = solve_reaching_definitions(textbook)
    check(inn[0] == set(), "nothing reaches the entry block")
    check((1, 0, "a") in inn[3] and (0, 1, "b") in inn[3] and (2, 0, "a") not in inn[3],
          "block 3 sees block 1's definitions only: %r" % (inn[3],))
    check(out[1] == {(0, 1, "b"), (1, 0, "a")}, "block 1 exports both of its definitions")
    check((1, 0, "a") in inn[2] and (0, 1, "b") in inn[2],
          "the loop body sees both the b definition and its own a definition")
    check(rounds > 1, "the loop forces more than one round (%d)" % rounds)
    oracle = reaching_oracle(textbook)
    check(all(inn[b] == oracle[b] for b in range(4)),
          "the oracle must reproduce every IN set: %r vs %r" % (inn, oracle))
    check(all(gen[b] <= out[b] for b in range(4)), "gen is always a subset of OUT")
    check(all(not (out[b] & (kill[b] - gen[b])) for b in range(4)),
          "a killed definition can only survive if it was generated in the same block")

    # ---- same fixpoint, different block order ---------------------------------------
    reverse = solve_reaching_definitions(textbook, order=[3, 2, 1, 0])
    check(reverse[0] == inn and reverse[1] == out,
          "the least fixpoint does not depend on the iteration order")
    check(reverse[2] <= 4 * rounds, "reverse order stays within a small round budget")

    # ---- round-by-round monotonicity -------------------------------------------------
    _, _, _, snapshots = solve_reaching_definitions(textbook, collect_rounds=True)
    check(len(snapshots) >= 2, "the snapshot mode must record every round")
    for earlier, later in zip(snapshots, snapshots[1:]):
        check(all(a <= b for a, b in zip(earlier[0], later[0])),
              "the IN sets must grow monotonically")
        check(all(a <= b for a, b in zip(earlier[1], later[1])),
              "the OUT sets must grow monotonically")

    # ---- the equations hold exactly at the fixpoint ----------------------------------
    residual = 0
    gen, kill = textbook.gen_kill()
    for b in range(len(textbook.blocks)):
        union_preds = set()
        for p in textbook.pred[b]:
            union_preds |= out[p]
        if union_preds != inn[b] or gen[b] | (inn[b] - kill[b]) != out[b]:
            residual += 1
    check(residual == 0, "the fixpoint must satisfy the transfer equations exactly")
    _, _, rounds_again, _ = solve_reaching_definitions(textbook)
    check(rounds_again == rounds, "re-solving must take the same number of rounds")

    # ---- randomized agreement with the path oracle -----------------------------------
    empty_in = 0
    for i in range(120):
        cfg = random_cfg(0x5EED + i * 7919, 3 + i % 4)
        got_in, got_out, got_rounds, _ = solve_reaching_definitions(cfg)
        check(all(got_in[b] == reaching_oracle(cfg)[b] for b in range(len(cfg.blocks))),
              "oracle mismatch on random CFG %d" % i)
        check(got_rounds <= len(cfg.definitions()) * len(cfg.blocks) + 1,
              "round count exceeded the theoretical bound on CFG %d" % i)
        for b in range(len(cfg.blocks)):
            union_preds = set()
            for p in cfg.pred[b]:
                union_preds |= got_out[p]
            check(union_preds == got_in[b], "IN must equal the predecessors' union (%d, %d)"
                  % (i, b))
            check(cfg.gen_kill()[0][b] <= got_out[b], "gen subset of OUT (%d, %d)" % (i, b))
            if not cfg.pred[b]:
                empty_in += got_in[b] == set()
    check(empty_in > 0, "entry-like blocks must have an empty IN set")

    # ---- boundary cases --------------------------------------------------------------
    single = Cfg([[("def", "x")]], [[]])
    check(solve_reaching_definitions(single)[2] == 2,
          "a single block converges after one full pass plus the confirming pass")
    check(solve_reaching_definitions(single)[1][0] == {(0, 0, "x")}, "the single def is live out")
    no_defs = Cfg([[("use", "z")], [("use", "z")]], [[1], []])
    check(solve_reaching_definitions(no_defs)[1] == [set(), set()],
          "a program that only reads variables has no definitions to propagate")
    unreachable = Cfg([[("def", "a")], [("def", "b")], [("use", "a")]], [[], [2], []])
    check(solve_reaching_definitions(unreachable)[0][1] == set(),
          "an unreachable block's IN set stays empty")
    check(solve_reaching_definitions(no_defs, order=[])[0] == [set(), set()],
          "an empty iteration order still yields the (trivial) least fixpoint")
    double = Cfg([[("def", "a"), ("def", "a")], [("use", "a")]], [[1], []])
    check(solve_reaching_definitions(double)[0][1] == {(0, 1, "a")},
          "only the second of two same-block definitions reaches the successor")

    # ---- error handling and determinism ----------------------------------------------
    for bad in (lambda: Cfg([[("def", "a")]], [[0]]),
                lambda: Cfg([[("def", "a")]], [[1]]),
                lambda: Cfg([[("def", "a")]], [])):
        rejected = False
        try:
            bad()
        except ValueError:
            rejected = True
        check(rejected, "malformed CFGs must be rejected")
    first = solve_reaching_definitions(random_cfg(12345))
    second = solve_reaching_definitions(random_cfg(12345))
    check(first[0] == second[0] and first[1] == second[1] and first[2] == second[2],
          "the analysis must be deterministic")

    print("textbook: %d rounds, %d definitions, loop IN has %d definitions"
          % (rounds, len(textbook.definitions()), len(inn[2])))
    print("random CFGs: 120/120 agreed with the path oracle")
    print("PASS")


if __name__ == "__main__":
    main()
