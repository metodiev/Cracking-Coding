"""Task 672 - Implement liveness analysis on small CFG and dead code elimination.  (Tier: Expert, Beyond — Compilers, Languages & Solvers)

Backward dataflow analysis plus the optimization it enables.

    use[B] = variables read before being written inside B (upward-exposed uses)
    def[B] = variables written in B
    IN[B]  = use[B] union (OUT[B] minus def[B])
    OUT[B] = union of IN[s] over the successors s

Liveness is the classic backward, union/must-not lattice: sets only grow, so the backward
iteration reaches a least fixpoint. A path-based per-variable oracle (a worklist that walks
predecessor edges, one variable at a time, stopping at blocks that redefine the variable)
cross-checks every IN set on 120 random CFGs.

Dead code elimination walks each block backwards from OUT[B]: a definition whose variable is
not live after it is deleted, and the pass is repeated until nothing changes, because
deleting one dead definition can make an earlier one dead too. Correctness is checked
semantically, on real executions: a deterministic interpreter runs 200 concrete paths
through the CFG and records every value observed by a `use`; the trace of the optimized
program must equal the trace of the original one, while the optimized program must be
strictly smaller on inputs that actually contain dead writes. Values depend only on the
variable and the block, so deleting an instruction cannot shift them.

Simplified: scalar variables only, no arrays or aliasing, and the CFG is given rather than
parsed. The interpreter observes use-sites only (plus the final environment), which is
exactly what dead code elimination must preserve.
"""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Cfg:
    """Control-flow graph: `blocks[i]` is a list of instructions, `succ[i]` successors.

    An instruction is ("def", var) or ("use", var).
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
                if t == b:
                    raise ValueError("self loops must be modelled with a real block")
        self.pred = [[] for _ in self.blocks]
        for b, targets in enumerate(self.succ):
            for t in targets:
                self.pred[t].append(b)

    def all_vars(self):
        return sorted({var for block in self.blocks for _, var in block})

    def use_def(self, b):
        """Upward-exposed uses and all definitions of block b."""
        used = set()
        defined = set()
        for op, var in self.blocks[b]:
            if op == "def":
                defined.add(var)
            elif var not in defined:
                used.add(var)
        return used, defined

    def instruction_count(self):
        return sum(len(block) for block in self.blocks)


def solve_liveness(cfg, order=None):
    """Backward fixpoint. Returns (inn, out, rounds)."""
    n = len(cfg.blocks)
    uses, defs = [], []
    for b in range(n):
        used, defined = cfg.use_def(b)
        uses.append(used)
        defs.append(defined)
    inn = [set() for _ in range(n)]
    out = [set() for _ in range(n)]
    order = list(range(n - 1, -1, -1)) if order is None else list(order)
    rounds = 0
    while True:
        rounds += 1
        changed = False
        for b in order:
            new_out = set()
            for s in cfg.succ[b]:
                new_out |= inn[s]
            new_in = uses[b] | (new_out - defs[b])
            if new_out != out[b] or new_in != inn[b]:
                out[b], inn[b] = new_out, new_in
                changed = True
        if not changed:
            return inn, out, rounds


def liveness_oracle(cfg):
    """Per-variable backward reachability: where is each variable live at block entries?"""
    n = len(cfg.blocks)
    live_in = [set() for _ in range(n)]
    for var in cfg.all_vars():
        worklist = []
        for b in range(n):
            used, _ = cfg.use_def(b)
            if var in used:
                live_in[b].add(var)
                worklist.append(b)
        while worklist:
            cur = worklist.pop()
            for p in cfg.pred[cur]:
                _, defined = cfg.use_def(p)
                if var not in defined and var not in live_in[p]:
                    live_in[p].add(var)
                    worklist.append(p)
    return live_in


def block_live_in(block, live_out):
    """Backward walk inside one block: returns live-at-entry and the positions that die."""
    live = set(live_out)
    dead_positions = []
    for k in range(len(block) - 1, -1, -1):
        op, var = block[k]
        if op == "use":
            live.add(var)
        elif var in live:
            live.discard(var)
        else:
            dead_positions.append(k)
    return live, sorted(dead_positions)


def dead_code_elimination(cfg, max_rounds=64):
    """Delete unobservable definitions until a fixpoint. Returns (cfg, removed)."""
    current = cfg
    removed_total = 0
    for _ in range(max_rounds):
        _, out, _ = solve_liveness(current)
        blocks = []
        removed = 0
        for b, block in enumerate(current.blocks):
            _, dead = block_live_in(block, out[b])
            dead = set(dead)
            removed += len(dead)
            blocks.append([ins for k, ins in enumerate(block) if k not in dead])
        current = Cfg(blocks, current.succ)
        removed_total += removed
        if removed == 0:
            return current, removed_total
    raise AssertionError("dead code elimination failed to converge")


def value_of(var, b):
    """Deterministic value produced by a definition - independent of the position in the
    block, so deleting other instructions cannot shift the observed values."""
    return (ord(var) - ord("a") + 1) * 1000 + b


def run_path(cfg, path):
    """Interpret a concrete path; returns the trace of values read by `use` instructions."""
    env = {}
    trace = []
    for b in path:
        for op, var in cfg.blocks[b]:
            if op == "def":
                env[var] = value_of(var, b)
            else:
                trace.append(env.get(var, 0))
    return trace


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return state, z ^ (z >> 31)


def random_cfg(seed, block_count=5, variables="abc"):
    """Random CFG: forward DAG plus one back edge, with a healthy mix of defs and uses."""
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
            op = "def" if raw % 2 else "use"
            instructions.append((op, variables[raw % len(variables)]))
        blocks.append(instructions)
    if block_count > 1:
        state, raw = splitmix64(state)
        succ[block_count - 1].append(raw % (block_count - 1))
    return Cfg(blocks, succ)


def random_paths(cfg, count, seed, max_length=20):
    paths = []
    state = seed
    for _ in range(count):
        path = [0]
        for _ in range(max_length):
            state, raw = splitmix64(state)
            targets = cfg.succ[path[-1]]
            if not targets:
                break
            path.append(targets[raw % len(targets)])
        paths.append(path)
    return paths


def main():
    # ---- a textbook liveness example -------------------------------------------------
    example = Cfg(
        [[("def", "a"), ("use", "b")],
         [("def", "b"), ("use", "a")],
         [("use", "c")],
         [("def", "c"), ("use", "a")]],
        [[1], [2, 3], [1], []])
    inn, out, rounds = solve_liveness(example)
    check(rounds > 1, "the loop forces several backward rounds (%d)" % rounds)
    check(inn[0] == {"b", "c"}, "the entry block is live-in for b and for c (which flows in "
          "through the loop) but not for a, which it defines first: %r" % (inn[0],))
    check(out[3] == set(), "the exit block has nothing live-out")
    check(inn[3] == {"a"}, "block 3 uses a before defining c: %r" % (inn[3],))
    check("a" in out[2] and "a" in inn[2], "liveness flows backwards through the loop: %r"
          % (inn[2],))
    oracle = liveness_oracle(example)
    check(all(inn[b] == oracle[b] for b in range(4)), "the per-variable oracle must agree")
    reverse = solve_liveness(example, order=[0, 1, 2, 3])
    check(reverse[0] == inn and reverse[1] == out,
          "the least fixpoint is independent of the iteration order")
    for b in range(4):
        used, defined = example.use_def(b)
        expected_in = used | (out[b] - defined)
        check(expected_in == inn[b], "IN must satisfy the transfer equation in block %d" % b)
        expected_out = set()
        for s in example.succ[b]:
            expected_out |= inn[s]
        check(expected_out == out[b], "OUT must be the union of the successors in block %d" % b)
        live_entry, _ = block_live_in(example.blocks[b], out[b])
        check(live_entry == inn[b], "the intra-block walk must reproduce IN in block %d" % b)

    # ---- randomized cross-checks -----------------------------------------------------
    removed_somewhere = 0
    for i in range(120):
        cfg = random_cfg(0xC0DE + i * 7919, 3 + i % 4)
        got_in, got_out, _ = solve_liveness(cfg)
        check(all(got_in[b] == liveness_oracle(cfg)[b] for b in range(len(cfg.blocks))),
              "oracle mismatch on random CFG %d" % i)
        for b in range(len(cfg.blocks)):
            used, defined = cfg.use_def(b)
            check(used <= got_in[b], "exposed uses are always live-in (%d, %d)" % (i, b))
            check(got_in[b] <= set(cfg.all_vars()) and got_out[b] <= set(cfg.all_vars()),
                  "no invented variables (%d, %d)" % (i, b))
        optimized, removed = dead_code_elimination(cfg)
        if removed:
            removed_somewhere += 1
        check(removed >= 0 and optimized.instruction_count() == cfg.instruction_count() - removed,
              "the removed count must match the size difference on CFG %d" % i)
        check(dead_code_elimination(optimized)[1] == 0,
              "the optimized CFG must be free of dead definitions (CFG %d)" % i)
        paths = random_paths(cfg, 3, 0x999 + i)
        for path in paths:
            check(run_path(cfg, path) == run_path(optimized, path),
                  "dead code elimination changed a program trace on CFG %d: %r" % (i, path))
    check(removed_somewhere > 20, "the sweep must really delete code (%d CFGs)" % removed_somewhere)

    # ---- targeted dead code ----------------------------------------------------------
    dead = Cfg([[("def", "x"), ("def", "y"), ("use", "x")], [("use", "x")]], [[1], []])
    optimized, removed = dead_code_elimination(dead)
    check(removed == 1, "only the never-read definition of y may go (%d)" % removed)
    check(optimized.blocks[0] == [("def", "x"), ("use", "x")],
          "the surviving instructions keep their order: %r" % (optimized.blocks[0],))

    per_path = Cfg([[("def", "x"), ("def", "y")], [("use", "y")], [("use", "x")]],
                   [[1, 2], [], []])
    kept, removed_both = dead_code_elimination(per_path)
    check(removed_both == 0, "definitions used on some path are live (%d removed)" % removed_both)
    check(kept.instruction_count() == per_path.instruction_count(), "nothing may disappear")

    chain = Cfg([[("def", "x"), ("def", "x"), ("def", "x")], [("use", "z")]], [[1], []])
    chained, removed_chain = dead_code_elimination(chain)
    check(removed_chain == 3, "three unobservable writes to x must all go (%d)" % removed_chain)
    check(chained.blocks[0] == [], "the block becomes empty: %r" % (chained.blocks[0],))

    loop_live = Cfg([[("def", "a")], [("use", "a")]], [[1], [0]])
    check(dead_code_elimination(loop_live)[1] == 0,
          "a definition that is read on the next loop iteration must survive")
    loop_dead = Cfg([[("def", "a")], [("use", "b"), ("def", "a")]], [[1], [0]])
    stale, removed_stale = dead_code_elimination(loop_dead)
    check(removed_stale == 2 and stale.blocks[0] == [],
          "writes to a never-read variable go, even inside a loop (%d)" % removed_stale)

    # ---- boundary cases --------------------------------------------------------------
    only_uses = Cfg([[("use", "p")], [("use", "q")]], [[1], []])
    live_uses, out_uses, _ = solve_liveness(only_uses)
    check(live_uses[0] == {"p", "q"} and live_uses[1] == {"q"},
          "reads without writes are live-in, and q is live in block 0 too: %r" % (live_uses,))
    check(dead_code_elimination(only_uses)[1] == 0, "uses are never deleted")
    trivial = Cfg([[]], [[]])
    check(solve_liveness(trivial)[0] == [set()], "an empty block has an empty IN set")
    check(dead_code_elimination(trivial)[0].instruction_count() == 0, "nothing to delete")
    unreachable = Cfg([[("def", "u")], [("use", "u")]], [[], []])
    check(solve_liveness(unreachable)[1][0] == set(),
          "an unreachable block has nothing live-out")
    self_use = Cfg([[("def", "s"), ("use", "s")]], [[]])
    check(solve_liveness(self_use)[0] == [set()],
          "a def-then-use pair leaves nothing live at the block entry")

    # ---- error handling and determinism ----------------------------------------------
    for bad in (lambda: Cfg([[("def", "a")]], [[0]]),
                lambda: Cfg([[("def", "a")]], [[2]]),
                lambda: Cfg([[("def", "a")]], [])):
        rejected = False
        try:
            bad()
        except ValueError:
            rejected = True
        check(rejected, "malformed CFGs must be rejected")
    first = solve_liveness(random_cfg(31337))
    second = solve_liveness(random_cfg(31337))
    check(first == second, "the analysis must be deterministic")
    check(dead_code_elimination(random_cfg(31337))[1] ==
          dead_code_elimination(random_cfg(31337))[1],
          "the optimization must be deterministic")

    print("example: %d round(s), exit IN = %r, loop IN = %r" % (rounds, sorted(inn[3]), sorted(inn[2])))
    print("random CFGs: 120/120 oracle agreement, %d CFGs contained dead definitions"
          % removed_somewhere)
    print("PASS")


if __name__ == "__main__":
    main()
