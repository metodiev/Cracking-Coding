"""Task 752 - Implement taint analysis: simple propagation for security checks.  (Tier: Expert, Beyond - Security & Privacy)

Taint analysis is a forward dataflow analysis over a program in SSA form.  Every variable holds a
*taint set*: the set of source labels whose value may have reached it.  Sources are seeded by
calls to a fixed allowlist of input functions (`read_cookie`, `argv`, `read_body`); the transfer
function is

    assign v = lit            -> {}                      assign v = var w      -> taint[w]
    assign v = call f(args)   -> {}                       if f is in SANITIZERS  (kill)
                              -> {label(f)} | U taint(args) if f is in SOURCES
                              -> U taint(args)             otherwise (unknown call: propagates)
    phi   v = (w1, w2, ...)   -> U taint[wi]              (join of the incoming edges)

and a *sink* statement (`sql_query`, `os_exec`, `write_file`) reports a flow whenever its operand
carries a label.  Sanitizers are an explicit allowlist: anything else - including a lookalike
`validate_weak` - propagates, so the analysis errs towards reporting.

The analysis is a worklist fixpoint over a def-use ("readers") index: a statement is re-queued
only when one of the variables it reads got a strictly larger taint set, which is exactly the
chaotic-iteration optimisation of the classic algorithm.  The framework is monotone (transfer
functions are unions only) and its domain is a finite lattice of label sets, so the least fixpoint
exists and the worklist terminates.  Two independent oracles are used in the checks: (1) a naive
implementation that sweeps every statement until nothing changes, and (2) a *backward* traversal
that starts at the sink's operand and walks def-use edges against the data flow, collecting source
labels and stopping at sanitizer calls.  Because the transfer functions are pure unions, backward
reachability must return exactly the same label set as the forward fixpoint - any disagreement is
a bug in the readers index or the kill rule.

Simplified versus a production taint analyser: the IR is a flat SSA statement list (no C/Java
parsing, no aliasing, no field sensitivity, no array element tracking), sanitisers kill the whole
taint instead of the argument they wrap, and the analysis is *path-insensitive* - the last check
below demonstrates the resulting false positive at a guarded sink.  Implicit flows (control flow
dependent on a secret) are not modelled, which is also stated as a known limitation.
"""
from collections import deque

MASK64 = (1 << 64) - 1

SOURCES = {"read_cookie": "cookie", "argv": "argv", "read_body": "body"}
SANITIZERS = {"escape_html", "int_cast", "sql_escape"}
SINKS = {"sql_query", "os_exec", "write_file"}


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


class ProgramError(ValueError):
    """Raised for malformed IR: duplicate definitions, undefined reads, bad statements."""


def reads_of(expr, into):
    """Variables read by an expression (recursive, three shape of nodes only)."""
    if not isinstance(expr, tuple) or not expr:
        raise ProgramError("malformed expression %r" % (expr,))
    if expr[0] == "var":
        into.append(expr[1])
    elif expr[0] == "lit":
        return into
    elif expr[0] == "call":
        for arg in expr[2]:
            reads_of(arg, into)
    else:
        raise ProgramError("unknown expression node %r" % (expr[0],))
    return into


class Program:
    """Flat SSA statement list: ('assign', v, expr) | ('phi', v, (vars,)) | ('sink', name, v)."""

    def __init__(self, statements):
        self.statements = [tuple(s) for s in statements]
        self.defs = {}
        for index, stmt in enumerate(self.statements):
            if stmt[0] == "assign":
                self._define(stmt[1], index)
            elif stmt[0] == "phi":
                if len(stmt) != 3 or not stmt[2]:
                    raise ProgramError("a phi node needs at least one incoming variable")
                self._define(stmt[1], index)
            elif stmt[0] == "sink":
                if stmt[1] not in SINKS:
                    raise ProgramError("unknown sink %r" % (stmt[1],))
            else:
                raise ProgramError("unknown statement kind %r" % (stmt[0],))
        self.readers = {var: [] for var in self.defs}
        for index, stmt in enumerate(self.statements):
            for var in self.operands(stmt):
                if var not in self.defs:
                    raise ProgramError("read of undefined variable %r" % var)
                self.readers[var].append(index)

    def _define(self, var, index):
        if var in self.defs:
            raise ProgramError("SSA violation: %r defined twice" % var)
        self.defs[var] = index

    def operands(self, stmt):
        if stmt[0] == "assign":
            return reads_of(stmt[2], [])
        if stmt[0] == "phi":
            return list(stmt[2])
        return [stmt[2]]

    def sinks(self):
        return [i for i, stmt in enumerate(self.statements) if stmt[0] == "sink"]


def expr_taint(expr, taint):
    """Transfer function for a single expression: pure union, plus the source/sanitizer rules."""
    if expr[0] == "var":
        return taint[expr[1]]
    if expr[0] == "lit":
        return frozenset()
    args = frozenset().union(*(expr_taint(a, taint) for a in expr[2])) if expr[2] else frozenset()
    func = expr[1]
    if func in SANITIZERS:                    # an allowlisted sanitizer removes the whole taint
        return frozenset()
    if func in SOURCES:
        return args | {SOURCES[func]}
    return args


def transfer(stmt, taint):
    """Taint assigned by a statement (sinks define nothing)."""
    if stmt[0] == "assign":
        return expr_taint(stmt[2], taint)
    if stmt[0] == "phi":
        return frozenset().union(*(taint[arg] for arg in stmt[2]))
    return frozenset()


def analyze(program):
    """Worklist forward analysis.  Returns taint sets, reported flows and iteration statistics."""
    taint = {var: frozenset() for var in program.defs}
    work = deque(range(len(program.statements)))
    queued = set(work)
    updates = 0
    passes = 0
    while work:
        index = work.popleft()
        queued.discard(index)
        passes += 1
        stmt = program.statements[index]
        if stmt[0] == "sink":
            continue
        fresh = transfer(stmt, taint)
        if fresh != taint[stmt[1]]:
            taint[stmt[1]] = fresh
            updates += 1
            for reader in program.readers[stmt[1]]:
                if reader not in queued:
                    queued.add(reader)
                    work.append(reader)
    flows = []
    for index in program.sinks():
        stmt = program.statements[index]
        labels = taint[stmt[2]]
        if labels:
            flows.append((index, stmt[1], stmt[2], tuple(sorted(labels))))
    return {"taint": taint, "flows": flows, "updates": updates, "steps": passes}


def naive_analyze(program):
    """Oracle 1: no readers index, just sweep every statement until nothing changes."""
    taint = {var: frozenset() for var in program.defs}
    rounds = 0
    while True:
        rounds += 1
        changed = False
        for stmt in program.statements:
            if stmt[0] == "sink":
                continue
            fresh = transfer(stmt, taint)
            if fresh != taint[stmt[1]]:
                taint[stmt[1]] = fresh
                changed = True
        if not changed:
            return taint, rounds


def backward_labels(program, operand):
    """Oracle 2: walk def-use edges backwards from a sink operand, collecting source labels.

    At a sanitizer call the walk stops (the kill rule) - nothing below it can taint the result.
    Cycles (loop phis) are handled by a visited set exactly like graph reachability.
    """
    labels = set()
    seen = set()
    stack = [operand]
    while stack:
        var = stack.pop()
        if var in seen:
            continue
        seen.add(var)
        index = program.defs.get(var)
        if index is None:
            continue
        stmt = program.statements[index]
        if stmt[0] == "phi":
            stack.extend(stmt[2])
        else:
            expr = stmt[2]
            if expr[0] == "call" and expr[1] in SANITIZERS:
                continue
            if expr[0] == "call" and expr[1] in SOURCES:
                labels.add(SOURCES[expr[1]])
            stack.extend(reads_of(expr, []))
    return frozenset(labels)


def random_program(state, count):
    """Random SSA program: atoms, source/sanitizer calls, joins and sinks, plus loop back-edges."""
    state, _ = splitmix64(state)
    statements = []
    names = []
    for i in range(count):
        state, r = splitmix64(state)
        var = "v%02d" % i
        pick = r % 10
        if pick < 3 or not names:
            state, r = splitmix64(state)
            if r % 3 == 0:
                expr = ("lit", 0)
            elif names:
                expr = ("var", names[r % len(names)])
            else:
                expr = ("call", "read_body", ())
        elif pick < 6:
            state, r = splitmix64(state)
            func = ("argv", "read_cookie", "read_body")[r % 3]
            args = ()
            if names and r % 2:
                state, r = splitmix64(state)
                args = (("var", names[r % len(names)]),)
            expr = ("call", func, args)
        elif pick < 7:
            state, r = splitmix64(state)
            func = ("escape_html", "int_cast", "validate_weak")[r % 3]
            args = ()
            if names:
                state, r = splitmix64(state)
                args = (("var", names[r % len(names)]),)
            expr = ("call", func, args)
        elif pick < 9 and names:
            picks = []
            for _ in range(1 + r % 3):
                state, r = splitmix64(state)
                picks.append(names[r % len(names)])
            expr = ("phi", tuple(picks))
            statements.append(("phi", var, expr[1]))
            names.append(var)
            continue
        else:
            expr = ("var", names[r % len(names)])
        statements.append(("assign", var, expr))
        names.append(var)
    # inject loop back-edges: some phis read variables defined later, which forces real iterations
    for index, stmt in enumerate(statements):
        state, r = splitmix64(state)
        if stmt[0] == "phi" and r % 3 == 0:
            state, r = splitmix64(state)
            statements[index] = ("phi", stmt[1], stmt[2] + (names[r % len(names)],))
    for index in range(0, count, 7):
        state, r = splitmix64(state)
        target = names[(index * 3 + r) % len(names)]
        statements.append(("sink", ("sql_query", "os_exec", "write_file")[r % 3], target))
    return statements, state


def all_flows(program):
    """Convenience: labels reported per sink, used by the oracle comparisons."""
    result = analyze(program)
    return {(stmt_index, var): labels for stmt_index, name, var, labels in result["flows"]}


def main():
    # 1. a straight source -> sink chain is reported with the exact source label
    chain = Program([
        ("assign", "a", ("call", "read_cookie", ())),
        ("assign", "b", ("call", "concat", (("var", "a"), ("lit", "x")))),
        ("sink", "sql_query", "b"),
    ])
    result = analyze(chain)
    check([f[3] for f in result["flows"]] == [("cookie",)],
          "direct flow must report exactly the cookie label")
    check(backward_labels(chain, "b") == frozenset({"cookie"}), "backward oracle must agree")

    # 2. allowlisted sanitizers kill the taint, while the unsanitized sibling stays tainted
    guarded = Program([
        ("assign", "raw", ("call", "argv", ())),
        ("assign", "clean", ("call", "escape_html", (("var", "raw"),))),
        ("assign", "dirty", ("call", "concat", (("var", "raw"), ("var", "clean")))),
        ("sink", "write_file", "clean"),
        ("sink", "os_exec", "dirty"),
    ])
    guarded_result = analyze(guarded)
    check(guarded_result["taint"]["clean"] == frozenset(), "sanitizer output must be clean")
    check(len(guarded_result["flows"]) == 1 and guarded_result["flows"][0][3] == ("argv",),
          "exactly the unsanitized sink must be reported")
    check(backward_labels(guarded, "dirty") == frozenset({"argv"}),
          "backward walk must not walk through the sanitizer")

    # 3. only the allowlist sanitizes: a lookalike validator leaves the flow visible
    lookalike = Program([
        ("assign", "raw", ("call", "read_body", ())),
        ("assign", "maybe", ("call", "validate_weak", (("var", "raw"),))),
        ("sink", "sql_query", "maybe"),
    ])
    check(analyze(lookalike)["flows"] and backward_labels(lookalike, "maybe") == frozenset({"body"}),
          "an unknown 'validator' must not be trusted as a sanitizer")

    # 4. joins: a phi merges labels, and the union of two sources is reported
    joined = Program([
        ("assign", "a", ("call", "argv", ())),
        ("assign", "b", ("call", "read_cookie", ())),
        ("phi", "c", ("a", "b")),
        ("sink", "os_exec", "c"),
    ])
    check(analyze(joined)["flows"][0][3] == ("argv", "cookie"), "phi must union both labels")

    # 5. boundary cases: empty program, no sources, literal sink, undefined reads rejected
    empty = Program([])
    check(analyze(empty)["flows"] == [] and analyze(empty)["updates"] == 0, "empty program is clean")
    literal = Program([("sink", "sql_query", "n"), ("assign", "n", ("lit", 7))])
    check(analyze(literal)["flows"] == [], "a constant cannot carry a taint label")
    for bad in ([("assign", "x", ("call", "f", (("var", "missing"),)))],
                [("sink", "sql_query", "ghost")],
                [("assign", "v", ("lit", 1)), ("assign", "v", ("lit", 2))],
                [("phi", "p", ("lit", 1))]):
        try:
            Program(bad)
            raise AssertionError("malformed IR must be rejected: %r" % (bad,))
        except ProgramError:
            pass

    # 6. oracle 1 - the naive sweep reaches the same least fixpoint on every random program
    state = 0x7A17
    programs = []
    for trial in range(30):
        statements, state = random_program(state, 24)
        program = Program(statements)
        programs.append(program)
        worklist = analyze(program)
        naive, rounds = naive_analyze(program)
        check(worklist["taint"] == naive,
              "worklist and naive sweep disagree on program %d" % trial)
        check(rounds >= 1, "the naive sweep must run at least one round")

    # 7. oracle 2 - backward reachability equals the forward fixpoint for every sink
    for trial, program in enumerate(programs):
        for index in program.sinks():
            stmt = program.statements[index]
            forward = analyze(program)["taint"][stmt[2]]
            check(forward == backward_labels(program, stmt[2]),
                  "forward/backward disagree at sink %d of program %d" % (index, trial))

    # 8. the results really are a least fixpoint, and loop back-edges make it non-trivial
    for program in programs:
        outcome = analyze(program)
        for stmt in program.statements:                 # applying the transfer again changes nothing
            if stmt[0] != "sink":
                check(transfer(stmt, outcome["taint"]) == outcome["taint"][stmt[1]],
                      "the reported state must be a fixpoint")
    looped = [analyze(p) for p in programs]
    check(any(o["steps"] > len(p.statements) for p, o in zip(programs, looped)),
          "at least one statement must be re-processed: loop back-edges force real iterations")

    # 9. monotonicity: forcing a value to be a source can only add labels
    base = Program([
        ("assign", "a", ("call", "clean_fn", ())),
        ("assign", "b", ("call", "concat", (("var", "a"), ("lit", "!")))),
        ("sink", "os_exec", "b"),
    ])
    tainted = Program([
        ("assign", "a", ("call", "read_body", ())),
        ("assign", "b", ("call", "concat", (("var", "a"), ("lit", "!")))),
        ("sink", "os_exec", "b"),
    ])
    base_taint = analyze(base)["taint"]
    tainted_taint = analyze(tainted)["taint"]
    check(base_taint["b"] == frozenset() and tainted_taint["b"] == frozenset({"body"}),
          "adding a source must be observable at the sink")
    check(all(base_taint[v] <= tainted_taint[v] for v in base_taint), "taint must be monotone")

    # 10. determinism and idempotence
    check(all_flows(programs[0]) == all_flows(Program(programs[0].statements)),
          "analysis must be deterministic")
    again = analyze(programs[3])
    check(again["updates"] == analyze(Program(programs[3].statements))["updates"],
          "the worklist update count must be reproducible")

    # 11. known limitation: path-insensitivity reports a false positive at a guarded sink
    guarded_sink = Program([
        ("assign", "a", ("call", "read_body", ())),
        ("assign", "safe", ("call", "int_cast", (("var", "a"),))),
        ("phi", "merged", ("safe", "a")),
        ("sink", "sql_query", "merged"),
    ])
    check(analyze(guarded_sink)["flows"],
          "the join of a cleaned and a raw value must still be reported (over-approximation)")

    total_sinks = sum(len(p.sinks()) for p in programs)
    total_flows = sum(len(analyze(p)["flows"]) for p in programs)
    print("752 random programs=%d sinks=%d reported flows=%d"
          % (len(programs), total_sinks, total_flows))
    print("752 worklist updates max=%d, naive rounds max=%d"
          % (max(o["updates"] for o in looped),
             max(naive_analyze(p)[1] for p in programs)))
    print("752 allowlisted sanitizers=%s, sinks=%s, sources=%s"
          % (len(SANITIZERS), len(SINKS), len(SOURCES)))
    print("PASS")


if __name__ == "__main__":
    main()
