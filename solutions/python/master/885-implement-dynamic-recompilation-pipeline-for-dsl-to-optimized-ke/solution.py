"""Task 885 - Implement dynamic recompilation pipeline for DSL to optimized kernels with heuristics.  (Tier: Master, CTCI Beyond - Compilers, Languages & Solvers)

The DSL is a "map" language: a program is a list of independent statements
`out_j = expr` evaluated for every index `i` of a column set, where an expression is built
from integer constants, element reads `load(col)[i]`, and the operators +, -, *.  Programs
are lowered to flat register code (one loop with fused stores), which is exactly what a
JIT emits after loop fusion.

The pipeline implements three execution tiers.  Tier 0 is a tree-walking interpreter that
also serves as the ground-truth oracle.  Tier 1 is the baseline register kernel produced by
the lowering pass (constant folding, global value numbering / CSE, loop fusion).  Tier 2 is
the speculative kernel: after profiling finds columns that were observed uniform (equal in
every row) across the invocation history, their loads are hoisted out of the loop and a
*guard* re-checks uniformity on every later invocation.  A guard failure is a deoptimizing
bailout: the pipeline immediately falls back to the baseline kernel (so the returned value
is still correct), blacklists the failed column, and recompiles with the reduced assumption
set - the standard "compile, guard, bail out, recompile" loop of a dynamic compiler.  The
recompile budget is explicit and bounded; once exhausted the program stays on the baseline
kernel forever, so compilation cannot loop indefinitely.

Heuristics: promotion is driven by invocation counters (hot threshold) and the choice of
which optimizations to apply is decided by a small cost model (per-op, per-read, per-guard
costs).  The heuristic plan is compared against an exhaustive search over all optimization
plans on small programs.

Simplified versus a production JIT: there is no machine code, no register allocator beyond
the naive one, no real profiling counters (invocation counts stand in), guards are checked
once per invocation rather than per basic block, the DSL has no control flow, loops or
memory side effects, and the cost model is analytic rather than measured.  The invariants
that are nevertheless verified: every tier produces exactly the interpreter's result
(differential testing with an independent stack-machine evaluator as a second oracle), the
guard is load-bearing (bypassing it on a non-uniform column demonstrably changes the
result), the recompile budget bounds the number of compilations, specialization strictly
reduces the modelled cost and the executed element reads, the greedy heuristic plan attains
the exhaustive optimum, and all runs are reproducible from the seed.
"""
from itertools import combinations

MASK64 = (1 << 64) - 1

BINARY_OPS = ("add", "sub", "mul")

OP_COST = 1        # one register operation
READ_COST = 4      # one element read from a column (memory access is expensive)
ITER_COST = 1      # loop bookkeeping per index
FILL_COST = 1      # storing a broadcast constant into one output element
GUARD_COST = 3     # re-checking one speculatively hoisted column, per element


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
    """Deterministic splitmix64 stream - the only source of randomness."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def u64(self):
        self.state, value = splitmix64(self.state)
        return value

    def below(self, bound):
        return self.u64() % bound

    def choice(self, items):
        return items[self.below(len(items))]


class GuardFailure(Exception):
    """Raised by a speculative kernel when one of its assumptions stopped holding."""

    def __init__(self, column):
        super().__init__("speculation about column %r no longer holds" % (column,))
        self.column = column


# --------------------------------------------------------------------- DSL
def const(value):
    return ("const", value)


def load(column):
    return ("load", column)


def binop(op, left, right):
    if op not in BINARY_OPS:
        raise ValueError("unknown operator %r" % (op,))
    return (op, left, right)


def make_program(statements, length):
    return {"statements": [(target, expr) for target, expr in statements], "length": length}


def validate_expression(expression, columns):
    kind = expression[0]
    if kind == "const":
        if type(expression[1]) is not int:
            raise ValueError("constant operands must be integers")
        return
    if kind == "load":
        if expression[1] not in columns:
            raise ValueError("unknown column %r" % (expression[1],))
        return
    if kind in BINARY_OPS:
        validate_expression(expression[1], columns)
        validate_expression(expression[2], columns)
        return
    raise ValueError("unknown node %r" % (kind,))


def validate_program(program, columns):
    seen = set()
    for target, expression in program["statements"]:
        if not target or not isinstance(target, str):
            raise ValueError("statement targets must be non-empty strings")
        if target in seen:
            raise ValueError("duplicate statement target %r" % (target,))
        seen.add(target)
        validate_expression(expression, columns)
    if program["length"] < 0:
        raise ValueError("loop length must be non-negative")
    return program


def columns_of(program):
    found = set()

    def walk(expression):
        kind = expression[0]
        if kind == "load":
            found.add(expression[1])
        elif kind in BINARY_OPS:
            walk(expression[1])
            walk(expression[2])

    for _, expression in program["statements"]:
        walk(expression)
    return found


def is_uniform(values):
    return len(set(values)) <= 1


def is_constant_expression(expression):
    return expression[0] == "const"


# --------------------------------------------------- oracle 1: tree interpreter
def eval_scalar(expression, env, index):
    kind = expression[0]
    if kind == "const":
        return expression[1]
    if kind == "load":
        return env[expression[1]][index]
    left = eval_scalar(expression[1], env, index)
    right = eval_scalar(expression[2], env, index)
    if kind == "add":
        return left + right
    if kind == "sub":
        return left - right
    return left * right


def interpret(program, env):
    """Ground truth: straightforward per-index evaluation of every statement."""
    results = {target: [] for target, _ in program["statements"]}
    for index in range(program["length"]):
        for target, expression in program["statements"]:
            results[target].append(eval_scalar(expression, env, index))
    return results


# --------------------------------- oracle 2: independent stack-machine evaluator
def to_postfix(expression, tokens=None):
    """Post-order token list; a separate implementation path from eval_scalar."""
    if tokens is None:
        tokens = []
    kind = expression[0]
    if kind in ("const", "load"):
        tokens.append(expression)
        return tokens
    to_postfix(expression[1], tokens)
    to_postfix(expression[2], tokens)
    tokens.append((kind,))
    return tokens


def eval_postfix(tokens, env, index):
    stack = []
    for token in tokens:
        kind = token[0]
        if kind == "const":
            stack.append(token[1])
        elif kind == "load":
            stack.append(env[token[1]][index])
        else:
            right = stack.pop()
            left = stack.pop()
            if kind == "add":
                stack.append(left + right)
            elif kind == "sub":
                stack.append(left - right)
            else:
                stack.append(left * right)
    return stack[0]


# ------------------------------------------------------------------- lowering
def fold_expression(expression):
    kind = expression[0]
    if kind in ("const", "load"):
        return expression
    left = fold_expression(expression[1])
    right = fold_expression(expression[2])
    if left[0] == "const" and right[0] == "const":
        return ("const", eval_scalar((kind, left, right), {}, 0))
    return (kind, left, right)


def compile_ir(program, fold=True, cse=True, invariants=()):
    """Lower a program to register code executed inside a single fused loop.

    Registers are slots in one flat file per iteration.  A speculatively hoisted load
    becomes a scalar register that is preset once per invocation and therefore costs no
    per-iteration instruction; the guard that justifies it is emitted separately.
    """
    invariants = frozenset(invariants)
    unknown = invariants - columns_of(program)
    if unknown:
        raise ValueError("cannot specialize on absent columns %s" % (sorted(unknown),))

    ops = []
    stores = []
    fills = []
    scalars = {}
    memo = {}
    next_register = [0]

    def new_register():
        index = next_register[0]
        next_register[0] += 1
        return index

    def emit(expression):
        if cse and expression in memo:
            return memo[expression]
        kind = expression[0]
        if kind == "const":
            dest = new_register()
            ops.append(("const", dest, expression[1]))
        elif kind == "load":
            column = expression[1]
            if column in invariants:
                if column not in scalars:
                    scalars[column] = new_register()      # preset before the loop
                dest = scalars[column]
            else:
                dest = new_register()
                ops.append(("load", dest, column))
        else:
            left = emit(expression[1])
            right = emit(expression[2])
            dest = new_register()
            ops.append((kind, dest, left, right))
        if cse:
            memo[expression] = dest
        return dest

    for target, expression in program["statements"]:
        folded = fold_expression(expression) if fold else expression
        if is_constant_expression(folded):
            fills.append((target, folded[1]))            # loop-invariant, computed once
        else:
            stores.append((target, emit(folded)))

    return {
        "ops": ops,
        "stores": stores,
        "fills": fills,
        "scalar_registers": {column: index for column, index in scalars.items()},
        "guarded_columns": tuple(sorted(scalars)),
        "register_count": next_register[0],
        "loop_ops": len(ops),
        "inline_reads": sum(1 for op in ops if op[0] == "load"),
        "prelude_reads": len(scalars),
    }


def estimate_cost(program, fold=True, cse=True, invariants=()):
    """Analytic per-invocation cost used by the optimization heuristics."""
    ir = compile_ir(program, fold=fold, cse=cse, invariants=invariants)
    length = program["length"]
    cost = length * (ir["loop_ops"] * OP_COST + ITER_COST + ir["inline_reads"] * READ_COST)
    cost += ir["prelude_reads"] * (READ_COST + OP_COST)
    cost += length * FILL_COST * len(ir["fills"])
    cost += length * GUARD_COST * len(ir["guarded_columns"])
    return cost


def heuristic_plan(program, candidate_invariants=()):
    """Greedy plan: accept each optimization exactly when it lowers the modelled cost."""
    plan = {"fold": False, "cse": False, "invariants": ()}
    best = estimate_cost(program, fold=False, cse=False, invariants=())

    trial = dict(plan, fold=True)
    if estimate_cost(program, **trial) < best:
        plan, best = trial, estimate_cost(program, **trial)
    trial = dict(plan, cse=True)
    if estimate_cost(program, **trial) < best:
        plan, best = trial, estimate_cost(program, **trial)

    for column in sorted(candidate_invariants):
        trial = dict(plan, invariants=plan["invariants"] + (column,))
        cost = estimate_cost(program, **trial)
        if cost < best:
            plan, best = trial, cost
    return plan


def exhaustive_plan(program, candidate_invariants=()):
    """Brute force over every (fold, cse, invariant subset) combination."""
    candidates = sorted(candidate_invariants)
    best_plan, best_cost = None, None
    for fold in (False, True):
        for cse in (False, True):
            for size in range(len(candidates) + 1):
                for subset in combinations(candidates, size):
                    cost = estimate_cost(program, fold=fold, cse=cse, invariants=subset)
                    if best_cost is None or cost < best_cost:
                        best_plan = {"fold": fold, "cse": cse, "invariants": subset}
                        best_cost = cost
    return best_plan, best_cost


# --------------------------------------------------------------------- runtime
class Kernel:
    """Executable form of one lowering at one specialization level."""

    def __init__(self, program, ir):
        self.program = program
        self.ir = ir
        self.guarded_columns = tuple(ir["guarded_columns"])
        self.invocations = 0
        self.element_reads = 0
        self.guard_scans = 0
        self.ops_executed = 0

    def run(self, env, guard=True):
        if guard:
            for column in self.guarded_columns:
                values = env[column]
                first = values[0] if values else None
                self.guard_scans += len(values)
                for value in values:
                    if value != first:
                        raise GuardFailure(column)
        return self.execute(env)

    def execute(self, env):
        """Unchecked execution - used by the pipeline after deoptimization."""
        length = self.program["length"]
        out = {target: [] for target, _ in self.program["statements"]}
        for target, value in self.ir["fills"]:
            out[target] = [value] * length
            self.ops_executed += 1

        registers = [0] * self.ir["register_count"]
        for column, index in self.ir["scalar_registers"].items():
            registers[index] = env[column][0] if length else 0

        ops = self.ir["ops"]
        stores = self.ir["stores"]
        for index in range(length):
            for op in ops:
                kind = op[0]
                if kind == "load":
                    registers[op[1]] = env[op[2]][index]
                elif kind == "add":
                    registers[op[1]] = registers[op[2]] + registers[op[3]]
                elif kind == "sub":
                    registers[op[1]] = registers[op[2]] - registers[op[3]]
                elif kind == "mul":
                    registers[op[1]] = registers[op[2]] * registers[op[3]]
                else:
                    registers[op[1]] = op[2]
            for target, register in stores:
                out[target].append(registers[register])

        self.invocations += 1
        self.element_reads += length * self.ir["inline_reads"]
        self.ops_executed += length * len(ops)
        return out


class RecompilationPipeline:
    """Tiered execution with profile-guided speculation, guards and a recompile budget."""

    INTERPRETER = 0
    BASELINE = 1
    SPECIALIZED = 2

    def __init__(self, programs, hot_threshold=3, max_recompiles=2):
        self.programs = dict(programs)
        self.hot_threshold = hot_threshold
        self.max_recompiles = max_recompiles
        self.kernels = {}
        self.compiles = 0
        self.cache_hits = 0
        self.guard_failures = 0
        self.state = {
            key: {"tier": self.INTERPRETER, "invocations": 0, "profile": frozenset(),
                  "blacklist": set(), "recompiles": 0, "deopts": 0,
                  "baseline": None, "specialized": None, "history": []}
            for key in self.programs
        }

    def kernel_for(self, key, invariants=()):
        """Cached compilation; the cache key is (program, specialization)."""
        entry = (key, tuple(sorted(invariants)))
        kernel = self.kernels.get(entry)
        if kernel is None:
            program = self.programs[key]
            ir = compile_ir(program, invariants=entry[1])
            kernel = Kernel(program, ir)
            self.kernels[entry] = kernel
            self.compiles += 1
        else:
            self.cache_hits += 1
        return kernel

    def _try_specialize(self, key, env):
        state = self.state[key]
        uniform = frozenset(c for c in columns_of(self.programs[key]) if is_uniform(env[c]))
        state["profile"] = uniform
        candidates = uniform - state["blacklist"]
        if not candidates:
            state["specialized"] = None
            state["tier"] = self.BASELINE
            return
        state["specialized"] = self.kernel_for(key, candidates)
        state["tier"] = self.SPECIALIZED

    def _select_tier(self, key, env):
        state = self.state[key]
        if state["tier"] == self.INTERPRETER and state["invocations"] >= self.hot_threshold:
            if state["baseline"] is None:
                state["baseline"] = self.kernel_for(key, ())
            state["tier"] = self.BASELINE
        if (state["tier"] == self.BASELINE and state["specialized"] is None
                and state["invocations"] >= 2 * self.hot_threshold
                and state["recompiles"] <= self.max_recompiles):
            self._try_specialize(key, env)

    def execute(self, key, env, audit=True):
        state = self.state[key]
        state["invocations"] += 1
        self._select_tier(key, env)
        tier = state["tier"]
        used = tier
        try:
            if tier == self.INTERPRETER:
                result = interpret(self.programs[key], env)
            elif tier == self.BASELINE:
                result = state["baseline"].run(env)
            else:
                result = state["specialized"].run(env)
        except GuardFailure as failure:
            self.guard_failures += 1
            state["deopts"] += 1
            state["recompiles"] += 1
            state["blacklist"].add(failure.column)
            state["specialized"] = None
            state["tier"] = self.BASELINE
            result = state["baseline"].run(env)      # baseline is assumption-free
            used = self.BASELINE
            if state["recompiles"] <= self.max_recompiles:
                self._try_specialize(key, env)
        if audit:
            expected = interpret(self.programs[key], env)
            check(result == expected,
                  "tier %d disagreed with the interpreter for program %r" % (tier, key))
        state["history"].append(used)
        return result


# ----------------------------------------------------------------- generators
def random_expression(rng, columns, depth):
    if depth == 0 or rng.below(4) == 0:
        return load(rng.choice(columns)) if rng.below(3) else const(rng.below(9) - 4)
    return binop(rng.choice(BINARY_OPS),
                 random_expression(rng, columns, depth - 1),
                 random_expression(rng, columns, depth - 1))


def random_program(rng, columns, length, statements=3):
    entries = []
    for index in range(statements):
        entries.append(("out%d" % index, random_expression(rng, columns, 3)))
    return make_program(entries, length)


def random_env(rng, columns, length, uniform_columns=()):
    env = {}
    for column in columns:
        if column in uniform_columns:
            value = rng.below(9) - 4
            env[column] = [value] * length
        else:
            env[column] = [rng.below(9) - 4 for _ in range(length)]
    return env


# ----------------------------------------------------------------------- main
def main():
    columns = ("a", "b", "c")

    # --- 1. the DSL rejects malformed programs -----------------------------
    raised = False
    try:
        validate_program(make_program([("out", load("zzz"))], 4), columns)
    except ValueError:
        raised = True
    check(raised, "a load from an unknown column must be rejected")
    raised = False
    try:
        binop("div", const(1), const(2))
    except ValueError:
        raised = True
    check(raised, "an unknown operator must be rejected")
    raised = False
    try:
        validate_program(make_program([("out", const(1)), ("out", const(2))], 4), columns)
    except ValueError:
        raised = True
    check(raised, "duplicate statement targets must be rejected")
    raised = False
    try:
        validate_program(make_program([("out", const(True))], 4), columns)
    except ValueError:
        raised = True
    check(raised, "non-integer constants must be rejected")

    rng = Rng(885001)

    # --- 2. two independent oracles agree ----------------------------------
    for _ in range(25):
        program = random_program(rng, columns, 1 + rng.below(6))
        env = random_env(rng, columns, program["length"])
        tree = interpret(program, env)
        stack = {target: [eval_postfix(to_postfix(expr), env, i)
                          for i in range(program["length"])]
                 for target, expr in program["statements"]}
        check(tree == stack,
              "the tree interpreter and the stack evaluator must agree on %r" % (program,))

    # --- 3. differential testing of the lowering ---------------------------
    trials = 0
    disagreements = 0
    for _ in range(30):
        program = random_program(rng, columns, 1 + rng.below(8))
        kernel = Kernel(program, compile_ir(program))
        for _ in range(3):
            env = random_env(rng, columns, program["length"])
            trials += 1
            if kernel.run(env) != interpret(program, env):
                disagreements += 1
    check(disagreements == 0,
          "%d of %d baseline executions disagreed with the interpreter" % (disagreements, trials))

    # --- 4. constant folding and CSE ---------------------------------------
    folded_program = make_program([("k", binop("mul", binop("add", const(2), const(3)), const(4)))], 8)
    folded_ir = compile_ir(folded_program)
    check(folded_ir["fills"] == [("k", 20)], "constant folding must evaluate 2+3 then *4")
    check(folded_ir["ops"] == [], "a constant statement must leave the loop body empty")
    check(Kernel(folded_program, folded_ir).run({}) == {"k": [20] * 8},
          "a folded statement must broadcast its value over the whole loop")

    repeated = make_program([("p", binop("add", binop("mul", load("a"), load("b")),
                                         binop("mul", load("a"), load("b"))))], 16)
    with_cse = compile_ir(repeated, cse=True)
    without_cse = compile_ir(repeated, cse=False)
    check(len(with_cse["ops"]) < len(without_cse["ops"]),
          "CSE must remove the duplicated product: %d vs %d ops"
          % (len(with_cse["ops"]), len(without_cse["ops"])))
    check(without_cse["inline_reads"] == 4 and with_cse["inline_reads"] == 2,
          "CSE must halve the redundant loads")
    env_repeated = {"a": [3] * 16, "b": [5] * 16}
    check(Kernel(repeated, with_cse).run(env_repeated) == Kernel(repeated, without_cse).run(env_repeated),
          "CSE must preserve semantics")

    # --- 5. speculating on a uniform column ---------------------------------
    program = make_program([("x", binop("add", load("a"), load("b"))),
                            ("y", binop("mul", load("a"), load("b")))], 64)
    varying_env = {"a": [i % 7 for i in range(64)], "b": [i % 5 for i in range(64)]}
    uniform_env = {"a": [i % 7 for i in range(64)], "b": [6] * 64}
    baseline = Kernel(program, compile_ir(program))
    speculative = Kernel(program, compile_ir(program, invariants=("b",)))
    baseline_result = baseline.run(varying_env)
    speculative_result = speculative.run(uniform_env)
    check(speculative_result == interpret(program, uniform_env),
          "the hoisted kernel must match the interpreter while the guard holds")
    check(speculative.element_reads < baseline.element_reads,
          "hoisting must reduce element reads: %d vs %d"
          % (speculative.element_reads, baseline.element_reads))
    raised = False
    try:
        speculative.run(varying_env)
    except GuardFailure as failure:
        raised = failure.column == "b"
    check(raised, "a violated assumption must raise a guard failure naming the column")
    unguarded = speculative.run(varying_env, guard=False)
    check(unguarded != interpret(program, varying_env),
          "hoisting a non-uniform column must really change the result (the guard is load bearing)")
    check(speculative.run(uniform_env) == speculative_result,
          "a guard that still holds must not disturb execution")
    check(baseline.run(varying_env) == baseline_result, "the baseline kernel must be deterministic")

    # --- 6. the cost model prefers the specialization -----------------------
    plain_cost = estimate_cost(program)
    speculative_cost = estimate_cost(program, invariants=("b",))
    check(speculative_cost < plain_cost,
          "specialization must lower modelled cost: %d vs %d" % (speculative_cost, plain_cost))

    # --- 7. greedy heuristic attains the exhaustive optimum -----------------
    worst_gap = 0
    for _ in range(25):
        candidate = random_program(rng, columns, 1 + rng.below(6))
        uniform_columns = sorted(c for c in columns_of(candidate) if rng.below(2))
        plan = heuristic_plan(candidate, uniform_columns)
        best_plan, best_cost = exhaustive_plan(candidate, uniform_columns)
        greedy_cost = estimate_cost(candidate, **plan)
        worst_gap = max(worst_gap, greedy_cost - best_cost)
        check(greedy_cost == best_cost,
              "greedy plan %r costs %d but the exhaustive optimum is %d (%r)"
              % (plan, greedy_cost, best_cost, best_plan))
        check(greedy_cost <= estimate_cost(candidate, fold=False, cse=False, invariants=()),
              "no plan may be more expensive than the unoptimized lowering")

    # --- 8. tier escalation under the hot threshold -------------------------
    pipeline = RecompilationPipeline({"k1": program}, hot_threshold=3, max_recompiles=2)
    for _ in range(9):
        pipeline.execute("k1", uniform_env, audit=True)
    check(pipeline.state["k1"]["history"] == [0, 0, 1, 1, 1, 2, 2, 2, 2],
          "promotion must follow the invocation counters, got %r"
          % (pipeline.state["k1"]["history"],))
    check(pipeline.compiles == 2, "baseline plus one specialized variant were expected, got %d"
          % pipeline.compiles)
    escalation_history = list(pipeline.state["k1"]["history"])

    # --- 9. guard failures deoptimize and the budget bounds recompilation ---
    hot_program = make_program([("x", binop("add", load("a"), load("b"))),
                                ("y", binop("sub", load("c"), load("a")))], 32)
    pipeline = RecompilationPipeline({"job": hot_program}, hot_threshold=2, max_recompiles=2)
    base = {"a": [1] * 32, "b": [2] * 32, "c": [3] * 32}
    history = []
    for _ in range(4):
        pipeline.execute("job", base, audit=True)          # all three columns uniform
    check(pipeline.state["job"]["tier"] == 2 and pipeline.guard_failures == 0,
          "the pipeline must be specialized while every column stays uniform")
    for broken in ("b", "c", "a"):
        env = dict(base)
        env[broken] = [index % 4 for index in range(32)]
        history.append((broken, pipeline.execute("job", env, audit=True), interpret(hot_program, env)))
    check(pipeline.state["job"]["history"] == [0, 1, 1, 2, 1, 1, 1],
          "every bailout invocation must run on the baseline, got %r"
          % (pipeline.state["job"]["history"],))
    check(pipeline.guard_failures == 3,
          "each violated assumption must bail out exactly once, got %d" % pipeline.guard_failures)
    check(pipeline.state["job"]["blacklist"] == {"a", "b", "c"},
          "failed assumptions must be blacklisted: %r" % (pipeline.state["job"]["blacklist"],))
    compiles_after_budget = pipeline.compiles
    check(compiles_after_budget == 4,
          "baseline + three speculative variants were expected, got %d" % compiles_after_budget)
    check(pipeline.state["job"]["tier"] == 1,
          "after the recompile budget is spent the program must stay on the baseline")
    for _ in range(6):
        pipeline.execute("job", base, audit=True)
    check(pipeline.compiles == compiles_after_budget,
          "compilation must stop once the budget is exhausted")
    check(all(result == expected for _, result, expected in history),
          "a deoptimized invocation must still return the interpreter's result")

    # --- 10. the kernel cache is keyed by (program, specialization) ---------
    cached = RecompilationPipeline({"k1": program, "k2": repeated})
    first = cached.kernel_for("k1", ("b",))
    before = cached.compiles
    check(cached.kernel_for("k1", ("b",)) is first, "an identical request must reuse the kernel")
    check(cached.kernel_for("k1", ("a",)) is not first, "a different plan must compile separately")
    check(cached.kernel_for("k2", ("b",)) is not first, "a different program must compile separately")
    check(cached.compiles == before + 2 and cached.cache_hits == 1,
          "compilation counters must track the cache: %d compiles, %d hits"
          % (cached.compiles, cached.cache_hits))
    raised = False
    try:
        cached.kernel_for("k1", ("nope",))
    except ValueError:
        raised = True
    check(raised, "specializing on a column the program never reads must be rejected")

    # --- 11. degenerate shapes ---------------------------------------------
    empty_program = make_program([("out", binop("add", load("a"), load("b")))], 0)
    check(Kernel(empty_program, compile_ir(empty_program)).run({"a": [], "b": []}) == {"out": []},
          "a zero-length loop must produce empty outputs")
    single = make_program([("out", binop("mul", load("a"), load("a")))], 1)
    check(Kernel(single, compile_ir(single)).run({"a": [7]}) == {"out": [49]},
          "a single-element program must be evaluated once")
    check(Kernel(single, compile_ir(single, invariants=("a",))).run({"a": [7]}) == {"out": [49]},
          "speculation must also work for a single-element program")

    # --- 12. exact arithmetic beyond 64 bits --------------------------------
    big = (1 << 62) + 12345
    big_program = make_program([("out", binop("mul", load("a"), const(big)))], 4)
    big_env = {"a": [big, big + 1, -big, 0]}
    check(Kernel(big_program, compile_ir(big_program)).run(big_env) == interpret(big_program, big_env),
          "wide integer arithmetic must be exact in every tier")

    # --- 13. reproducibility ------------------------------------------------
    def run_schedule(seed):
        local_rng = Rng(seed)
        envs = [random_env(local_rng, columns, 32, uniform_columns=("b", "c")) for _ in range(12)]
        local = RecompilationPipeline({"p": random_program(local_rng, columns, 32)},
                                      hot_threshold=2, max_recompiles=2)
        results = [local.execute("p", env, audit=True) for env in envs]
        return results, local.state["p"]["history"], local.compiles

    first_run = run_schedule(885500)
    second_run = run_schedule(885500)
    other_run = run_schedule(885501)
    check(first_run == second_run, "an identical seed must reproduce the whole schedule")
    check(first_run[1] == [0, 1, 1] + [2] * 9,
          "the recorded tier history must be stable, got %r" % (first_run[1],))
    check(other_run[1] == first_run[1] and other_run[2] == first_run[2],
          "a different workload must still follow the same promotion counters")

    # --- 14. larger sweep through the pipeline ------------------------------
    scale_program = random_program(Rng(885900), ("a", "b", "c", "d", "e"), 256, statements=4)
    scale_env = random_env(Rng(885901), ("a", "b", "c", "d", "e"), 256,
                           uniform_columns=("d", "e"))
    scale_pipeline = RecompilationPipeline({"big": scale_program}, hot_threshold=2)
    for _ in range(40):
        scale_pipeline.execute("big", scale_env, audit=True)
    check(scale_pipeline.state["big"]["history"][-1] == 2,
          "the large program must reach the specialized tier")
    check(scale_pipeline.guard_failures == 0,
          "uniform columns kept uniform must never trip a guard")

    print("differential trials: %d baseline executions, %d disagreements" % (trials, disagreements))
    print("greedy vs exhaustive plan: equal cost on 25 random programs (worst gap %d)" % worst_gap)
    print("cost model: baseline %d, specialized %d units per invocation" % (plain_cost, speculative_cost))
    print("tier history (hot threshold 3): %r" % (escalation_history,))
    print("speculative recompiles: %d, cache hits %d, guard failures %d"
          % (scale_pipeline.compiles, scale_pipeline.cache_hits, pipeline.guard_failures))
    print("PASS")


if __name__ == "__main__":
    main()
