"""Task 974 - Implement step-by-step solution explainer generator using AST of canonical solution.  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)

Each canonical solution is an abstract syntax tree of a tiny imperative language (assignments, while
loops, return) over expressions of numbers, variables and binary operators.  A tracing interpreter walks
the tree and records one event per executed node - a binding, a loop-condition test (with its iteration
and loop nesting depth) or the return - and the explainer turns that log into a numbered walkthrough
quoting each expression and the value it produced, renderable one step at a time.

Simplified versus a real explainer: the canonical solution is an AST literal rather than parsed source,
with no type inference and no data-flow slicing.  Verified: closed-form oracles (builtin sum, rows*cols)
and a second, differently structured interpreter, the exact trace composition, nested event counts against
a closed form, never-executed and empty programs, rejected unknown variables/operators and division by
zero, trace replay fidelity, drill-down and reproducibility.
"""
from collections import namedtuple

Event = namedtuple("Event", "kind detail value depth iteration")
SYMBOLS = {"add": "+", "sub": "-", "mul": "*", "floordiv": "//",
           "lt": "<", "gt": ">", "eq": "==", "le": "<=", "ge": ">="}
OPS = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
       "floordiv": lambda a, b: a // b, "lt": lambda a, b: a < b, "gt": lambda a, b: a > b,
       "eq": lambda a, b: a == b, "le": lambda a, b: a <= b, "ge": lambda a, b: a >= b}
# canonical solutions as ASTs: ("assign", name, expr) | ("while", cond, body) | ("return", expr)
SUM_PROGRAM = (
    ("assign", "total", ("num", 0)),
    ("assign", "index", ("num", 0)),
    ("while", ("lt", ("var", "index"), ("var", "limit")), (
        ("assign", "total", ("add", ("var", "total"), ("var", "index"))),
        ("assign", "index", ("add", ("var", "index"), ("num", 1))),
    )),
    ("return", ("var", "total")),
)
NESTED_PROGRAM = (
    ("assign", "hits", ("num", 0)), ("assign", "outer", ("num", 0)),
    ("while", ("lt", ("var", "outer"), ("var", "rows")), (
        ("assign", "inner", ("num", 0)),
        ("while", ("lt", ("var", "inner"), ("var", "cols")), (
            ("assign", "hits", ("add", ("var", "hits"), ("num", 1))),
            ("assign", "inner", ("add", ("var", "inner"), ("num", 1))))),
        ("assign", "outer", ("add", ("var", "outer"), ("num", 1))))),
    ("return", ("var", "hits")),
)
STRAIGHT_PROGRAM = (("assign", "answer", ("mul", ("num", 6), ("num", 7))), ("return", ("var", "answer")))
BAD_PROGRAMS = ((ValueError, (("return", ("var", "missing")),)),          # unknown variable
                (ValueError, (("assign", "value", ("mod", ("num", 7), ("num", 3))),)),  # unknown operator
                (ZeroDivisionError, (("assign", "q", ("floordiv", ("num", 10), ("num", 0))),)))


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def expect(error, call, message):
    try:
        call()
    except error:
        return
    raise AssertionError(message)


def evaluate(expr, env):
    """Primary evaluator: explicit dispatch over the AST expression nodes."""
    kind = expr[0]
    if kind == "num":
        return expr[1]
    if kind == "var":
        if expr[1] not in env:
            raise ValueError(f"unknown variable {expr[1]!r} in the canonical solution")
        return env[expr[1]]
    if kind not in OPS:
        raise ValueError(f"unknown operator {kind!r} in the canonical solution")
    left, right = evaluate(expr[1], env), evaluate(expr[2], env)
    if kind == "floordiv" and right == 0:
        raise ZeroDivisionError("division by zero while explaining the canonical solution")
    return OPS[kind](left, right)


def describe(expr):
    """Human-readable form of an expression, used in the test events and the explanations."""
    kind = expr[0]
    if kind == "num":
        return str(expr[1])
    if kind == "var":
        return expr[1]
    return f"{describe(expr[1])} {SYMBOLS[kind]} {describe(expr[2])}"


def run(program, env, trace, depth=0, iteration=0):
    """Execute the AST, appending one event per executed node; a return yields ("returned", value)."""
    for statement in program:
        kind = statement[0]
        if kind == "assign":
            value = evaluate(statement[2], env)
            env[statement[1]] = value
            trace.append(Event("assign", statement[1], value, depth, iteration))
        elif kind == "while":
            rounds = 0
            while True:
                outcome = evaluate(statement[1], env)
                trace.append(Event("test", describe(statement[1]), outcome, depth, rounds))
                if not outcome:
                    break
                if (outcome := run(statement[2], env, trace, depth + 1, rounds)) is not None:
                    return outcome
                rounds += 1
        elif kind == "return":
            value = evaluate(statement[1], env)
            trace.append(Event("return", "result", value, depth, iteration))
            return ("returned", value)
        else:
            raise ValueError(f"unknown statement {kind!r} in the canonical solution")
    return None


def interpret(program, env):
    """Independent oracle: a second interpreter that leaves the block with an exception - no tracing."""
    class Returned(Exception):
        pass

    environment = dict(env)

    def walk(block):
        for statement in block:
            kind = statement[0]
            if kind == "assign":
                environment[statement[1]] = evaluate(statement[2], environment)
            elif kind == "while":
                while evaluate(statement[1], environment):
                    walk(statement[2])
            elif kind == "return":
                environment["result"] = evaluate(statement[1], environment)
                raise Returned
            else:
                raise ValueError(f"unknown statement {kind!r} in the canonical solution")

    try:
        walk(program)
    except Returned:
        pass
    return environment


def explain(trace):
    """One numbered sentence per event: what ran, with the values that justify the step."""
    lines = []
    for number, event in enumerate(trace, start=1):
        if event.kind == "assign":
            lines.append(f"{number}. Step {number}: {event.detail} becomes {event.value} "
                         f"(loop level {event.depth}, iteration {event.iteration + 1}).")
        elif event.kind == "test":
            lines.append(f"{number}. Step {number}: the condition {event.detail} is {event.value}, so the loop "
                         f"{'continues' if event.value else 'stops'} "
                         f"(level {event.depth}, iteration {event.iteration + 1}).")
        else:
            lines.append(f"{number}. Step {number}: return {event.value}.")
    return lines


def explain_program(program, bindings=None):
    """Trace and explain one canonical solution, returning the walkthrough and the final state."""
    env, trace = dict(bindings or {}), []
    outcome = run(program, env, trace)
    return {"trace": trace, "env": env, "result": outcome[1] if outcome else None, "explanation": explain(trace)}


def explain_step(trace, number):
    """Render a single step of an already generated walkthrough (1-based, as shown to the learner)."""
    if not 1 <= number <= len(trace):
        raise IndexError(f"step {number} is outside the {len(trace)}-step walkthrough")
    return explain(trace)[number - 1]


def replay(trace):
    """Rebuild the final bindings from the event log alone - the log must be a faithful record."""
    env = {}
    for event in trace:
        if event.kind == "assign":
            env[event.detail] = event.value
    return env


def main():
    walkthrough = explain_program(SUM_PROGRAM, {"limit": 5})
    trace, explanation, env = walkthrough["trace"], walkthrough["explanation"], walkthrough["env"]
    # 1) happy path: the walkthrough computes the closed form (0+1+...+limit-1) and explains every step.
    check(walkthrough["result"] == sum(range(5)) == 10 and env == {"total": 10, "index": 5, "limit": 5}
          and len(explanation) == len(trace) and explanation[-1] == "19. Step 19: return 10."
          and all(line.startswith(f"{number}. Step {number}:") for number, line in enumerate(explanation, 1)),
          f"the sum walkthrough is wrong: {walkthrough['result']} / {explanation[-1]}")
    # 2) oracle: a second interpreter with a different control structure agrees on every program.
    for program, bindings, expected in ((SUM_PROGRAM, {"limit": 7}, 21), (SUM_PROGRAM, {"limit": 0}, 0),
                                        (NESTED_PROGRAM, {"rows": 4, "cols": 3}, 12),
                                        (STRAIGHT_PROGRAM, None, 42)):
        traced, oracle = explain_program(program, bindings), interpret(program, dict(bindings or {}))
        check(traced["env"] == {k: v for k, v in oracle.items() if k != "result"}
              and traced["result"] == oracle.get("result") == expected,
              f"the interpreter oracle disagrees on {program[0]}: {traced['env']} vs {oracle}")
    # 3) trace composition: setup binds first, one test per round plus the final exit test, return last.
    kinds, tests = [e.kind for e in trace], [e.value for e in trace if e.kind == "test"]
    check(kinds.count("assign") == 12 and kinds.count("test") == 6 and kinds.count("return") == 1
          and [event.detail for event in trace[:2]] == ["total", "index"] and kinds[-1] == "return"
          and tests == [True] * 5 + [False],
          f"the sum trace composition is wrong: {kinds.count('assign')}/{kinds.count('test')} {tests}")
    # 4) nested loops: event counts match the closed form and the nesting depth is recorded.
    nested = explain_program(NESTED_PROGRAM, {"rows": 4, "cols": 3})
    rows, cols = 4, 3
    assigns = 2 + rows * (2 + 2 * cols)   # setup, then per round: inner=0, hits+=1, inner+=1, outer+=1
    kinds_nested = [event.kind for event in nested["trace"]]
    check(nested["result"] == rows * cols and kinds_nested.count("assign") == assigns
          and kinds_nested.count("test") == (rows + 1) + rows * (cols + 1) == 21
          and max(event.depth for event in nested["trace"]) == 2
          and {event.depth for event in nested["trace"] if event.kind == "test"} == {0, 1}
          and {event.depth for event in nested["trace"] if event.kind == "return"} == {0},
          f"the nested-loop walkthrough is wrong: {nested['result']} {kinds_nested.count('assign')} assigns")
    # 5) boundaries: a loop that never runs, and a program with nothing in it.
    skipped, empty = explain_program(SUM_PROGRAM, {"limit": 0}), explain_program(())
    check([event.kind for event in skipped["trace"]] == ["assign", "assign", "test", "return"]
          and skipped["result"] == 0 and skipped["env"]["total"] == 0 and skipped["trace"][2].value is False
          and len(skipped["explanation"]) == 4 and empty["trace"] == [] and empty["explanation"] == []
          and empty["result"] is None and empty["env"] == {},
          f"the never-executed and empty boundaries are wrong: {skipped['trace']}")
    # 6) rejected inputs: unknown variable, unknown operator, division by zero, unknown statement.
    for error, program in BAD_PROGRAMS:
        expect(error, lambda program=program: explain_program(program), f"{error.__name__} must be raised")
    expect(ValueError, lambda: run((("nope",),), {}, []), "an unknown statement must be rejected")
    # 7) fidelity and drill-down: the log replays into the final bindings and any step can be rendered.
    check(replay(trace) == {name: value for name, value in env.items() if name != "limit"}
          and all(isinstance(event.value, int) and not isinstance(event.value, bool)
                  for event in trace if event.kind == "assign")
          and all(isinstance(event.value, bool) for event in trace if event.kind == "test")
          and [explain_step(trace, number) for number in range(1, len(trace) + 1)] == explanation
          and all(str(event.value) in explain_step(trace, number) for number, event in enumerate(trace, 1)),
          f"the trace is not a faithful record: {replay(trace)} vs {env}")
    for number, walk in ((len(trace) + 1, trace), (0, trace), (1, [])):
        expect(IndexError, lambda number=number, walk=walk: explain_step(walk, number), "step out of range")
    # 8) determinism: the whole walkthrough, its explanations and the oracle reproduce exactly.
    again = explain_program(SUM_PROGRAM, {"limit": 5})
    check(again["trace"] == trace and again["explanation"] == explanation and again["env"] == env
          and explain(nested["trace"]) == nested["explanation"]
          and interpret(NESTED_PROGRAM, {"rows": 4, "cols": 3}) == interpret(NESTED_PROGRAM, {"rows": 4, "cols": 3}),
          "the walkthrough must be reproducible")
    print(f"programs=4 steps={len(trace)} assigns={kinds.count('assign')} tests={kinds.count('test')} "
          f"result={walkthrough['result']} nested_steps={len(nested['trace'])} "
          f"first={explanation[0]!r} last={explanation[-1]!r}")
    print("PASS")


if __name__ == "__main__":
    main()
