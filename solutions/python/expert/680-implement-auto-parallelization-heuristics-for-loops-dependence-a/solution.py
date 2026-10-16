"""Task 680 - Implement auto-parallelization heuristics for loops (dependence analysis).  (Tier: Expert, Beyond — Compilers, Languages & Solvers)

A one-dimensional loop with affine array subscripts is analysed for dependences and then
actually executed in the schedules the analysis permits.

Analysis
  * every statement contributes accesses `array[a*i + b]`; a dependence exists when two
    accesses of the same array can touch the same element in different iterations, i.e.
    `a1*i1 + b1 == a2*i2 + b2` with i1 != i2 in [0, n);
  * the algebraic route is the classical gcd test - the equation `A*i1 + B*i2 = C` has a
    solution only when gcd(A, B) divides C - followed by an exact bounded scan over the
    one-parameter family of solutions that the extended Euclid produces, which turns the
    test into a precise dependence-distance computation;
  * an independent enumerative oracle simply checks every (i1, i2) pair, so the gcd
    reasoning is cross-checked on random subscript pairs;
  * the classification is DOALL (no loop-carried dependence), reduction (the only carried
    dependences are commutative accumulations into a scalar), DOACROSS (a bounded positive
    distance, i.e. a pipeline is legal) or SERIAL.

Execution
  * the sequential order is the reference; a naive "interleaved" parallel order shows that a
    non-parallelizable loop really does produce a different result, which is what justifies
    refusing it;
  * a DOALL loop must produce the identical result under any iteration order;
  * a reduction is executed with private per-worker accumulators and combined, and must
    equal the sequential sum exactly;
  * a DOACROSS loop is executed with a distance-d wavefront (residue classes of d run in
    parallel, each in order) and must also be exact.

Simplified: one loop level (no nests, no loop interchange), integer arithmetic only, and the
dependence test covers affine subscripts of a single induction variable.
"""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def extended_gcd(a, b):
    if b == 0:
        return abs(a), (1 if a >= 0 else -1), 0
    g, x, y = extended_gcd(b, a % b)
    return g, y, x - (a // b) * y


def solve_linear_range(a1, b1, a2, b2, length):
    """Solve a1*i1 + b1 == a2*i2 + b2 over i1, i2 in [0, length).

    Returns None when the gcd test rules out any solution, otherwise
    (carried, min_distance) where carried is True when a solution with i1 != i2 exists.
    The solution family is i1 = i1_0 + (B/g)*t, i2 = i2_0 - (A/g)*t, so a bounded scan over
    the t values that keep both indices in range is enough.
    """
    A, B, C = a1, -a2, b2 - b1
    if A == 0 and B == 0:
        if C != 0 or length < 2:
            return None
        return True, 1                     # every pair of iterations collides
    g, x, y = extended_gcd(A, B)
    if C % g != 0:
        return None
    scale = C // g
    i1_0, i2_0 = x * scale, y * scale
    step1, step2 = B // g, -(A // g)
    low, high = -10 ** 12, 10 ** 12
    for base, step in ((i1_0, step1), (i2_0, step2)):
        if step == 0:
            if not 0 <= base < length:
                return None
        elif step > 0:
            low = max(low, -(-(0 - base) // step))
            high = min(high, (length - 1 - base) // step)
        else:
            low = max(low, -(-(length - 1 - base) // step))
            high = min(high, (0 - base) // step)
    if low > high:
        return None
    best = None
    for t in range(low, high + 1):         # at most `length` values, not length**2
        i1, i2 = i1_0 + step1 * t, i2_0 + step2 * t
        if i1 != i2:
            distance = abs(i1 - i2)
            best = distance if best is None else min(best, distance)
    return (False, 0) if best is None else (True, best)


class Loop:
    """Statements are (dest, (da, db), op, src, (sa, sb), const) with

        set :  dest[da*i+db] = src[sa*i+sb] + const
        add :  dest[da*i+db] = dest[da*i+db] + const      (src is None)
        plus:  dest[da*i+db] = dest[da*i+db] + src[sa*i+sb] + const
        sub :  dest[da*i+db] = dest[da*i+db] - src[sa*i+sb] - const
    """

    OPS = ("set", "add", "plus", "sub")

    def __init__(self, length, statements, initial=None):
        if length < 1:
            raise ValueError("a loop needs at least one iteration")
        self.length = length
        self.statements = []
        for statement in statements:
            dest, (da, db), op = statement[0], statement[1], statement[2]
            src, coeffs, const = statement[3], statement[4], statement[5]
            if op not in self.OPS:
                raise ValueError("unknown statement operation %r" % (op,))
            if op != "add" and src is None:
                raise ValueError("operation %r needs a source array" % (op,))
            if op == "add" and src is not None:
                raise ValueError("the in-place increment takes no source array")
            self.statements.append(statement)
        self.arrays = sorted({name for statement in self.statements
                              for name in (statement[0], statement[3]) if name})
        self.initial = {name: {index: initial(name, index) if initial else 1
                               for index in range(-2, length + 2)}
                        for name in self.arrays}

    def accesses(self):
        for index, statement in enumerate(self.statements):
            dest, (da, db), op, src, coeffs, const = statement
            yield (dest, da, db, "write", index)
            if op in ("add", "plus", "sub"):
                yield (dest, da, db, "read", index)   # the read-modify-write of the destination
            if src is not None:
                yield (src, coeffs[0], coeffs[1], "read", index)

    def scalar_accumulators(self):
        return {statement[0] for statement in self.statements
                if statement[1] == (0, 0) and statement[2] in ("plus", "sub")}


def dependence_pairs(loop):
    """Enumerative oracle: every (write, other) collision, with its distance."""
    found = []
    writes = [a for a in loop.accesses() if a[3] == "write"]
    others = list(loop.accesses())
    for dest, da, db, _, write_statement in writes:
        for name, a, b, kind, statement in others:
            if name != dest or (kind == "write" and statement == write_statement):
                continue
            best = None
            for i1 in range(loop.length):
                for i2 in range(loop.length):
                    if i1 != i2 and da * i1 + db == a * i2 + b:
                        best = abs(i1 - i2) if best is None else min(best, abs(i1 - i2))
            if best is not None:
                found.append({"write_statement": write_statement, "kind": kind,
                              "statement": statement, "distance": best})
    return found


def algebraic_dependences(loop):
    """The heuristic: gcd test + bounded scan over the solution family."""
    found = []
    writes = [a for a in loop.accesses() if a[3] == "write"]
    others = list(loop.accesses())
    for dest, da, db, _, write_statement in writes:
        for name, a, b, kind, statement in others:
            if name != dest or (kind == "write" and statement == write_statement):
                continue
            answer = solve_linear_range(da, db, a, b, loop.length)
            if answer is None:
                continue
            carried, distance = answer
            if carried:
                found.append({"write_statement": write_statement, "kind": kind,
                              "statement": statement, "distance": distance})
    return found


def analyse(loop):
    """Classify the loop using the algebraic dependence information."""
    dependences = algebraic_dependences(loop)
    carried = [d for d in dependences if d["distance"] > 0]
    if not carried:
        return {"verdict": "DOALL", "dependences": dependences, "distance": 0}
    accumulators = loop.scalar_accumulators()
    if accumulators and all(d["write_statement"] in
                           {i for i, s in enumerate(loop.statements) if s[0] in accumulators}
                           for d in carried):
        return {"verdict": "reduction", "dependences": dependences, "distance": 0}
    distance = min(d["distance"] for d in carried)
    return {"verdict": "doacross", "dependences": dependences, "distance": distance}


def execute(loop, order):
    """Evaluate every iteration in the given order; returns the array state."""
    state = {name: dict(values) for name, values in loop.initial.items()}
    for i in order:
        for dest, (da, db), op, src, coeffs, const in loop.statements:
            d = da * i + db
            s = coeffs[0] * i + coeffs[1] if src is not None else None
            if op == "set":
                state[dest][d] = state[src].get(s, 0) + const
            elif op == "add":
                state[dest][d] = state[dest].get(d, 0) + const
            elif op == "plus":
                state[dest][d] = state[dest].get(d, 0) + state[src].get(s, 0) + const
            else:
                state[dest][d] = state[dest].get(d, 0) - state[src].get(s, 0) - const
    return state


def sequential_order(length):
    return list(range(length))


def parallel_order(length, workers):
    """The order a naive parallelizer could produce: workers claim every w-th iteration."""
    if workers < 1:
        raise ValueError("at least one worker is required")
    return [i for start in range(workers) for i in range(start, length, workers)]


def wavefront_order(length, distance):
    """Distance-d pipeline: iterations with the same residue modulo d run in parallel."""
    if distance < 1:
        raise ValueError("the wavefront distance must be positive")
    return [i for residue in range(distance) for i in range(residue, length, distance)]


def execute_reduction(loop, workers):
    """Private per-worker accumulators, combined afterwards. Only the accumulator statement
    is parallelized; everything else has already been proven sequential."""
    accumulator = sorted(loop.scalar_accumulators())[0]
    position = [i for i, s in enumerate(loop.statements) if s[0] == accumulator][0]
    _, _, op, src, coeffs, const = loop.statements[position]
    base = execute(loop, sequential_order(loop.length))
    partials = []
    stolen = parallel_order(loop.length, workers)
    for worker in range(workers):
        mine = [i for slot, i in enumerate(stolen) if slot % workers == worker]
        total = 0
        for i in mine:
            term = base[src].get(coeffs[0] * i + coeffs[1], 0) + const
            total = total + term if op == "plus" else total - term
        partials.append(total)
    result = {name: dict(values) for name, values in base.items()}
    result[accumulator][0] = loop.initial[accumulator].get(0, 0) + sum(partials)
    return result, partials


def difference(left, right):
    """The set of (array, index) entries where two executions disagree."""
    diff = set()
    for name in set(left) | set(right):
        for index in set(left.get(name, {})) | set(right.get(name, {})):
            if left.get(name, {}).get(index) != right.get(name, {}).get(index):
                diff.add((name, index))
    return diff


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return state, z ^ (z >> 31)


def random_access_pairs(count, seed, length=12, max_coeff=3):
    state = seed
    pairs = []
    for _ in range(count):
        state, raw = splitmix64(state)
        a1 = raw % (2 * max_coeff + 1) - max_coeff
        state, raw = splitmix64(state)
        b1 = raw % 9 - 4
        state, raw = splitmix64(state)
        a2 = raw % (2 * max_coeff + 1) - max_coeff
        state, raw = splitmix64(state)
        b2 = raw % 9 - 4
        pairs.append((a1, b1, a2, b2, length))
    return pairs


def oracle_pairs(coefficients):
    a1, b1, a2, b2, length = coefficients
    best = None
    for i1 in range(length):
        for i2 in range(length):
            if i1 != i2 and a1 * i1 + b1 == a2 * i2 + b2:
                best = abs(i1 - i2) if best is None else min(best, abs(i1 - i2))
    return None if best is None else (True, best)


def initial_value(name, index):
    return ((ord(name[0]) + 7 * index) % 11) + 1


def main():
    # ---- the gcd test against full enumeration ---------------------------------------
    disagreements = carried_count = 0
    for coefficients in random_access_pairs(200, 0xC0DE):
        algebraic = solve_linear_range(*coefficients)
        oracle = oracle_pairs(coefficients)
        algebraic_distance = None if algebraic is None or not algebraic[0] else algebraic[1]
        oracle_distance = None if oracle is None else oracle[1]
        if algebraic_distance != oracle_distance:
            disagreements += 1
        carried_count += oracle is not None
    check(disagreements == 0,
          "the gcd route and the pair oracle must agree (%d disagreements)" % disagreements)
    check(0 < carried_count < 200,
          "the sweep must contain both dependent and independent pairs (%d)" % carried_count)
    check(solve_linear_range(2, 0, 2, 1, 12) is None,
          "a[2i] versus a[2i+1]: gcd 2 does not divide the offset, so no dependence exists")
    check(solve_linear_range(2, 0, 2, 2, 12)[1] == 1,
          "a[2i] versus a[2i+2] collides only one iteration apart")
    check(solve_linear_range(0, 0, 0, 0, 12) == (True, 1),
          "the same scalar position in two different iterations always collides")

    # ---- classification ---------------------------------------------------------------
    doall = Loop(6, [("a", (1, 0), "add", None, None, 1)], initial=initial_value)
    carried = Loop(6, [("a", (1, 0), "set", "a", (1, -1), 1)], initial=initial_value)
    backward2 = Loop(6, [("a", (1, 0), "set", "a", (1, -2), 1)], initial=initial_value)
    distant = Loop(6, [("a", (1, 0), "set", "a", (1, 2), 0)], initial=initial_value)
    strided = Loop(6, [("a", (2, 0), "set", "a", (2, 1), 0)], initial=initial_value)
    reduction = Loop(6, [("s", (0, 0), "plus", "a", (1, 0), 0)], initial=initial_value)
    check(analyse(doall)["verdict"] == "DOALL", "a[i] = a[i] + 1 is a DOALL loop")
    check(analyse(strided)["verdict"] == "DOALL", "a[2i] = a[2i+1] is a DOALL loop")
    check(analyse(carried)["verdict"] == "doacross" and analyse(carried)["distance"] == 1,
          "a[i] = a[i-1] + 1 carries a dependence at distance 1: %r" % (analyse(carried),))
    check(analyse(backward2)["verdict"] == "doacross" and analyse(backward2)["distance"] == 2,
          "a[i] = a[i-2] + 1 carries a dependence at distance 2: %r" % (analyse(backward2),))
    check(analyse(distant)["verdict"] == "doacross" and analyse(distant)["distance"] == 2,
          "a[i] = a[i+2] carries a dependence at distance 2: %r" % (analyse(distant),))
    check(analyse(reduction)["verdict"] == "reduction",
          "s = s + a[i] is a reduction: %r" % (analyse(reduction),))
    for loop in (doall, carried, backward2, distant, strided, reduction):
        check(algebraic_dependences(loop) == dependence_pairs(loop),
              "the algebraic analysis must match the enumerative oracle for %r"
              % (loop.statements,))

    # ---- execution: DOALL is order independent ---------------------------------------
    base = execute(doall, sequential_order(doall.length))
    for workers in (1, 2, 3, 4, 6):
        order = parallel_order(doall.length, workers)
        check(sorted(order) == sequential_order(doall.length),
              "a parallel order must be a permutation of all iterations")
        check(execute(doall, order) == base,
              "a DOALL loop must give the same result in any order (%d workers)" % workers)
    strided_base = execute(strided, sequential_order(strided.length))
    check(execute(strided, parallel_order(strided.length, 3)) == strided_base,
          "a[2i] = a[2i+1] is order independent")

    # ---- execution: the refusal is justified -----------------------------------------
    carried_base = execute(carried, sequential_order(carried.length))
    naive = execute(carried, parallel_order(carried.length, 2))
    check(difference(carried_base, naive),
          "a distance-1 carried dependence must produce a different result when reordered")
    check(execute(carried, wavefront_order(carried.length, 1)) == carried_base,
          "the distance-1 wavefront must reproduce the sequential result")
    two_base = execute(backward2, sequential_order(backward2.length))
    check(difference(two_base, execute(backward2, parallel_order(backward2.length, 3))),
          "a distance-2 dependence also breaks under a 3-worker block schedule")
    check(execute(backward2, wavefront_order(backward2.length, 2)) == two_base,
          "the distance-2 wavefront must reproduce the sequential result")
    distant_base = execute(distant, sequential_order(distant.length))
    check(execute(distant, wavefront_order(distant.length, 2)) == distant_base,
          "the distance-2 wavefront is also valid for the forward dependence")

    # ---- execution: reductions ------------------------------------------------------
    reduction_base = execute(reduction, sequential_order(reduction.length))
    for workers in (1, 2, 3, 6):
        combined, partial = execute_reduction(reduction, workers)
        check(combined["s"][0] == reduction_base["s"][0],
              "the private-accumulator reduction must equal the sequential sum (%d workers)"
              % workers)
        check(reduction.initial["s"][0] + sum(partial) == reduction_base["s"][0],
              "the initial value plus the partial sums must equal the total (%d workers)"
              % workers)
        check(len(partial) == workers, "one partial sum per worker")
    check(execute(reduction, parallel_order(reduction.length, 4))["s"][0]
          == reduction_base["s"][0],
          "integer addition is commutative, so even the naive order sums the same total")

    # ---- boundary cases, errors and determinism --------------------------------------
    single = Loop(1, [("a", (1, 0), "add", None, None, 1)], initial=initial_value)
    check(analyse(single)["verdict"] == "DOALL", "a single-iteration loop cannot carry anything")
    check(execute(single, [0])["a"][0] == initial_value("a", 0) + 1,
          "a length-one loop still executes its body")
    subtract = Loop(4, [("d", (1, 0), "sub", "d", (1, -1), 1)], initial=initial_value)
    check(analyse(subtract)["verdict"] == "doacross", "subtraction carries a dependence too")
    rejected = 0
    for bad in (lambda: Loop(0, []),
                lambda: Loop(4, [("a", (1, 0), "frob", "a", (1, 0), 0)]),
                lambda: Loop(4, [("a", (1, 0), "set", None, (1, 0), 0)]),
                lambda: Loop(4, [("a", (1, 0), "add", "a", (1, 0), 0)]),
                lambda: parallel_order(4, 0),
                lambda: wavefront_order(4, 0)):
        try:
            bad()
        except ValueError:
            rejected += 1
    check(rejected == 6, "invalid loops and schedules must be rejected (%d of 6)" % rejected)
    check(solve_linear_range(1, 0, 1, -1, 8) == solve_linear_range(1, 0, 1, -1, 8),
          "the analysis must be deterministic")
    check(execute(carried, sequential_order(carried.length)) == carried_base,
          "execution must be deterministic")

    print("random subscript pairs: 200/200 gcd route agrees with enumeration (%d dependent)"
          % carried_count)
    print("verdicts: %s, %s(d=%d), %s(d=%d), %s"
          % (analyse(doall)["verdict"], analyse(carried)["verdict"], analyse(carried)["distance"],
             analyse(distant)["verdict"], analyse(distant)["distance"],
             analyse(reduction)["verdict"]))
    print("reordering the carried loop changes %d entries" % len(difference(carried_base, naive)))
    print("PASS")


if __name__ == "__main__":
    main()
