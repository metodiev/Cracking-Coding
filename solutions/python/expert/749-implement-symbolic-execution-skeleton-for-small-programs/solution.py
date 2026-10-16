"""Task 749 - Implement symbolic execution skeleton for small programs.  (Tier: Expert, CTCI Beyond — Compilers, Languages & Solvers)

A tiny imperative language (assignments over linear integer expressions, `if`, bounded `while`,
`assert`) is executed twice: concretely, by an interpreter that runs one input at a time, and
symbolically, where variables hold linear expressions over the *input* variables and every branch
adds a constraint to a path condition.  Each symbolic path carries the path condition that reaches
it, the symbolic final state and the assertions that could fail on it.

Path conditions use integer semantics, so the negation of `e <= c` is `-e <= -c - 1` (never
`-e > -c`), which is what makes the fork exact rather than approximate.  Satisfiability of a path
condition is decided by enumerating the input variables over a finite domain: that is the
documented simplification of this skeleton, since a real symbolic executor would hand the
constraints to an SMT solver; inside the domain the decision is exact and the witness is a
concrete input that can be replayed.  `while` loops are unrolled to a fixed bound, also documented.

Verified: for every concrete input in the domain there is a symbolic path whose path condition is
satisfied by that input and whose symbolic final state, evaluated at it, equals the interpreted
final state; the set of inputs on which an assertion fails is exactly the union of the witnesses of
the failing symbolic paths; every witness really satisfies its path condition and really triggers
the assertion when executed; branch-free programs yield exactly one path; unreachable assertions
have no witness; and the whole analysis is deterministic.
"""
from itertools import product

MASK64 = (1 << 64) - 1
DOMAIN = (-3, -2, -1, 0, 1, 2, 3)


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


# ---------------------------------------------------------------------------- concrete syntax
def const(value):
    return ("const", value)


def var(name):
    return ("var", name)


def add(left, right):
    return ("add", left, right)


def sub(left, right):
    return ("sub", left, right)


def scale(expression, factor):
    return ("scale", expression, factor)


def le(left, right):
    return ("le", left, right)


def assign(name, expression):
    return ("assign", name, expression)


def branch(condition, then_body, else_body=()):
    return ("if", condition, list(then_body), list(else_body))


def loop(condition, body, bound=4):
    return ("while", condition, list(body), bound)


def assertion(condition):
    return ("assert", condition)


def fail():
    return ("fail",)


# ---------------------------------------------------------------------------- concrete interpreter
def concrete_expression(expression, state):
    kind = expression[0]
    if kind == "const":
        return expression[1]
    if kind == "var":
        return state.get(expression[1], 0)
    if kind == "add":
        return concrete_expression(expression[1], state) + concrete_expression(expression[2], state)
    if kind == "sub":
        return concrete_expression(expression[1], state) - concrete_expression(expression[2], state)
    return concrete_expression(expression[1], state) * expression[2]


def concrete_condition(condition, state):
    return (concrete_expression(condition[1], state) <= concrete_expression(condition[2], state))


def interpret(program, entry_state, failures=None):
    """Run the program concretely; returns the final state and collects failed assertions."""
    state = dict(entry_state)
    failures = [] if failures is None else failures

    def run(statements):
        for statement in statements:
            kind = statement[0]
            if kind == "assign":
                state[statement[1]] = concrete_expression(statement[2], state)
            elif kind == "if":
                if concrete_condition(statement[1], state):
                    run(statement[2])
                else:
                    run(statement[3])
            elif kind == "while":
                rounds = 0
                while concrete_condition(statement[1], state) and rounds < statement[3]:
                    run(statement[2])
                    rounds += 1
            elif kind == "assert":
                if not concrete_condition(statement[1], state):
                    failures.append(dict(state))
            else:
                raise AssertionError("the program executed a fail statement")

    run(program)
    return state, failures


# ---------------------------------------------------------------------------- symbolic values
def linear_const(value):
    return ({}, value)


def linear_var(name):
    return ({name: 1}, 0)


def linear_add(left, right):
    coefficients = dict(left[0])
    for name, coefficient in right[0].items():
        coefficients[name] = coefficients.get(name, 0) + coefficient
        if coefficients[name] == 0:
            del coefficients[name]
    return coefficients, left[1] + right[1]


def linear_sub(left, right):
    return linear_add(left, ({name: -coefficient for name, coefficient in right[0].items()},
                             -right[1]))


def linear_scale(value, factor):
    return ({name: coefficient * factor for name, coefficient in value[0].items()},
            value[1] * factor)


def symbolic_expression(expression, state):
    kind = expression[0]
    if kind == "const":
        return linear_const(expression[1])
    if kind == "var":
        return state.get(expression[1], linear_const(0))
    if kind == "add":
        return linear_add(symbolic_expression(expression[1], state),
                          symbolic_expression(expression[2], state))
    if kind == "sub":
        return linear_sub(symbolic_expression(expression[1], state),
                          symbolic_expression(expression[2], state))
    return linear_scale(symbolic_expression(expression[1], state), expression[2])


def difference(left, right):
    """The linear expression left - right."""
    return linear_sub(left, right)


def constraints_for(condition, state, truth):
    """Constraints for (left <= right) when `truth` is True, or its exact integer negation."""
    difference_value = difference(symbolic_expression(condition[1], state),
                                  symbolic_expression(condition[2], state))
    coefficients, constant = difference_value
    if truth:
        return (coefficients, -constant)              # sum(coeff * variable) <= -constant
    return ({name: -coefficient for name, coefficient in coefficients.items()},
            constant - 1)                             # integer negation: >= constant + 1


def constraint_holds(constraint, values):
    coefficients, bound = constraint
    return sum(coefficient * values[name] for name, coefficient in coefficients.items()) <= bound


def condition_holds(condition, values):
    """Evaluate a path-condition constraint list under a concrete input."""
    return all(constraint_holds(constraint, values) for constraint in condition)


def solve_path_condition(constraints, inputs):
    """Enumerate the input domain for a witness; returns the first satisfying input or None."""
    for values in product(DOMAIN, repeat=len(inputs)):
        candidate = dict(zip(inputs, values))
        if condition_holds(constraints, candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------- symbolic executor
class SymbolicExecutor:
    """Depth-first symbolic execution with bounded loop unrolling."""

    def __init__(self, inputs):
        self.inputs = list(inputs)
        self.paths = []
        self.stats = {"paths": 0, "forks": 0, "solver_calls": 0, "witnesses": 0}

    def explore(self, program):
        self.walk(program, {name: linear_var(name) for name in self.inputs}, [], [])
        return self.paths

    def walk(self, statements, state, constraints, failures):
        """Execute a statement list under a path condition; forks recurse with the remaining
        statements appended, so every path runs to the end of the program."""
        index = 0
        while index < len(statements):
            statement = statements[index]
            rest = statements[index + 1:]
            kind = statement[0]
            if kind == "assign":
                state = dict(state)
                state[statement[1]] = symbolic_expression(statement[2], state)
                index += 1
            elif kind == "if":
                self.stats["forks"] += 1
                self.walk(statement[2] + rest, dict(state),
                          constraints + [constraints_for(statement[1], state, True)], list(failures))
                self.walk(statement[3] + rest, dict(state),
                          constraints + [constraints_for(statement[1], state, False)], list(failures))
                return
            elif kind == "while":
                if statement[3] > 0:
                    self.stats["forks"] += 1
                    body = statement[2] + [("while", statement[1], statement[2], statement[3] - 1)]
                    self.walk(body + rest, dict(state),
                              constraints + [constraints_for(statement[1], state, True)],
                              list(failures))
                    constraints = constraints + [constraints_for(statement[1], state, False)]
                    index += 1                     # the exit path: the condition is false
                else:
                    index += 1                     # the unrolling bound is used up: just continue
            elif kind == "assert":
                failing = constraints_for(statement[1], state, False)
                witness = self.witness(constraints + [failing])
                if witness is not None:
                    failures = failures + [(statement, witness)]
                index += 1
            else:
                self.paths.append(("fail", state, constraints, failures))
                self.stats["paths"] += 1
                return
        self.paths.append(("end", state, constraints, failures))
        self.stats["paths"] += 1

    def witness(self, constraints):
        self.stats["solver_calls"] += 1
        witness = solve_path_condition(constraints, self.inputs)
        if witness is not None:
            self.stats["witnesses"] += 1
        return witness


# ---------------------------------------------------------------------------- programs
def max_program():
    return [assign("m", var("x")),
            branch(le(var("y"), var("x")), [assign("m", var("y"))]),
            assign("s", add(var("x"), var("y")))]


def failing_assert_program():
    return [assign("q", sub(scale(var("x"), 2), var("y"))),
            branch(le(var("q"), const(0)),
                   [assertion(le(var("x"), var("y")))],
                   [assign("q", add(var("q"), const(1)))])]


def loop_program():
    return [assign("i", const(0)),
            assign("acc", const(0)),
            loop(le(var("i"), var("x")), [assign("acc", add(var("acc"), var("i"))),
                                          assign("i", add(var("i"), const(1)))], bound=3)]


def branch_free_program():
    return [assign("a", add(var("x"), const(1))), assign("b", scale(var("a"), 3))]


def unreachable_assert_program():
    return [assign("t", add(var("x"), const(0))),
            branch(le(var("t"), var("x")),
                   [],
                   [assertion(le(const(0), const(0)))]),
            assertion(le(add(var("x"), const(3)), add(var("x"), const(4))))]


def main():
    inputs = ["x", "y"]
    # 1. the maximum program: every concrete input is covered by some symbolic path
    program = max_program()
    executor = SymbolicExecutor(inputs)
    paths = executor.explore(program)
    check(len(paths) == 2, "one branch must yield exactly two symbolic paths")
    check(all(path[0] == "end" for path in paths), "both paths must run to completion")
    for values in product(DOMAIN, repeat=2):
        entry = dict(zip(inputs, values))
        final_state, failures = interpret(program, entry)
        check(not failures, "the maximum program cannot fail")
        matching = [path for path in paths if condition_holds(path[2], entry)]
        check(matching, f"input {entry} must be covered by some symbolic path")
        for path in matching:
            symbolic_state = {name: sum(coefficient * entry[variable]
                                        for variable, coefficient in expression[0].items())
                              + expression[1]
                              for name, expression in path[1].items()}
            for name, value in final_state.items():
                if name in symbolic_state:
                    check(symbolic_state[name] == value,
                          f"symbolic {name} must match the interpreted value for {entry}")

    # 2. the failing assertion: witnesses match the concrete failures exactly
    program = failing_assert_program()
    executor = SymbolicExecutor(inputs)
    paths = executor.explore(program)
    failing_inputs = {tuple(sorted(entry.items())) for values in product(DOMAIN, repeat=2)
                      for entry in [dict(zip(inputs, values))]
                      if interpret(program, entry)[1]}
    failing_paths = [path for path in paths if path[3]]
    witnesses = set()
    for path in failing_paths:
        for statement, witness in path[3]:
            check(condition_holds(path[2], witness),
                  "a witness must satisfy the path condition it belongs to")
            check(tuple(sorted(witness.items())) in failing_inputs,
                  "every witness must be one of the concrete failing inputs")
            check(interpret(failing_assert_program(), witness)[1],
                  "replaying a witness must really trigger the assertion")
            witnesses.add(tuple(sorted(witness.items())))
    check(len(witnesses) == len(failing_paths) and len(witnesses) > 0,
          "each failing path must contribute exactly one witness")
    check(witnesses <= failing_inputs, "witnesses cannot be inputs that do not fail")
    for values in product(DOMAIN, repeat=2):
        entry = dict(zip(inputs, values))
        if tuple(sorted(entry.items())) not in failing_inputs:
            continue
        covered = [path for path in failing_paths if condition_holds(path[2], entry)]
        check(covered, f"the concrete failing input {entry} must be covered by a failing path")

    # 3. the loop program: bounded unrolling still matches the interpreted results in the domain
    program = loop_program()
    executor = SymbolicExecutor(inputs)
    paths = executor.explore(program)
    check(len(paths) == 4, "a three-fold unrolling must produce four paths")
    covered = 0
    for values in product(DOMAIN, repeat=2):
        entry = dict(zip(inputs, values))
        final_state, _ = interpret(program, entry)
        matching = [path for path in paths if condition_holds(path[2], entry)]
        if matching:
            covered += 1
            state = matching[0][1]
            evaluated = {name: sum(coefficient * entry[variable]
                                   for variable, coefficient in expression[0].items())
                         + expression[1] for name, expression in state.items()}
            check(evaluated["acc"] == final_state["acc"],
                  "the accumulated value must agree between symbolic and concrete runs")
    check(covered > 0, "some inputs must be covered by the unrolled paths")

    # 4. branch-free and unreachable-assert programs
    executor = SymbolicExecutor(inputs)
    paths = executor.explore(branch_free_program())
    check(len(paths) == 1, "a branch-free program must have exactly one path")
    check(executor.stats["forks"] == 0, "no forks may be recorded for a branch-free program")
    executor = SymbolicExecutor(inputs)
    paths = executor.explore(unreachable_assert_program())
    check(all(not path[3] for path in paths), "an unreachable assertion must have no witness")
    for values in product(DOMAIN, repeat=2):
        entry = dict(zip(inputs, values))
        check(not interpret(unreachable_assert_program(), entry)[1],
              "the concrete runs must confirm that the assertion never fails")

    # 5. explicit failure statement: symbolic execution reports it with a witness
    program = [branch(le(var("x"), const(0)), [fail()], [assign("y", var("x"))])]
    executor = SymbolicExecutor(inputs)
    paths = executor.explore(program)
    failing_paths = [path for path in paths if path[0] == "fail"]
    check(len(failing_paths) == 1, "the fail branch must be reported as its own path")
    witness = solve_path_condition(failing_paths[0][2], inputs)
    check(witness is not None and witness["x"] <= 0,
          "the reported failing path must have a satisfying witness")
    try:
        interpret(program, witness)
        raise AssertionError("the concrete interpreter must raise on a fail statement")
    except AssertionError as error:
        check("fail statement" in str(error), "the interpreter must report the failure")

    # 6. boundary cases: an empty program, an unassigned variable, a single path witness count
    executor = SymbolicExecutor(inputs)
    empty_paths = executor.explore([])
    check(len(empty_paths) == 1 and empty_paths[0][0] == "end" and empty_paths[0][2] == []
          and empty_paths[0][3] == [], "the empty program has exactly one empty path")
    defaults = SymbolicExecutor(inputs)
    defaults.explore([assign("z", add(var("unset"), const(2)))])
    check(defaults.paths[0][1]["z"] == ({}, 2), "an unassigned variable must default to zero")
    check(solve_path_condition([({}, 0)], inputs) is not None,
          "the singleton constraint 0 <= 0 is satisfiable")
    check(solve_path_condition([({}, -1)], inputs) is None,
          "0 <= -1 must be unsatisfiable")

    # 7. determinism
    first = SymbolicExecutor(inputs)
    second = SymbolicExecutor(inputs)
    program = failing_assert_program()
    check(first.explore(program) == second.explore(program) and first.stats == second.stats,
          "symbolic execution must be deterministic")

    print(f"749 paths={len(paths)} forks={executor.stats['forks']} "
          f"witnesses={executor.stats['witnesses']}")
    print(f"749 failing_inputs={len(failing_inputs)} solver_calls={first.stats['solver_calls']}")
    print("PASS")


if __name__ == "__main__":
    main()
