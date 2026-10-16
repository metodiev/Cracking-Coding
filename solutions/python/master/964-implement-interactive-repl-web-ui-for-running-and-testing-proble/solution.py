"""Task 964 - Implement interactive REPL/web UI for running and testing problems in-browser.
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

REPL model: a browser client posts request objects to a session; the session parses the command
(`let name = expr`, a bare expression, `:reset`, `:help`), evaluates it on a small recursive-descent
interpreter and answers with a response object echoing the request id plus the value, the step count
and - on failure - an error code and token position.  Evaluation runs inside a step budget, so
`sum(1, 1000000)` is stopped instead of freezing the tab, and nothing is committed until evaluation
succeeds, which makes every request atomic.  Simplified: integers with `+ - * / %`, parentheses,
variables and the looping `sum(a, b)` primitive; no DOM, the "network" is a direct method call.
Verified: arithmetic against Python's own evaluation as an oracle, variable persistence, budget
enforcement and atomicity, protocol ids, parse positions, and the empty/over-long/reset boundaries.
"""
import re

TOKEN_RE = re.compile(r"\d+|[A-Za-z_][A-Za-z_0-9]*|[+\-*/%(),=]")
DEFAULT_BUDGET = 300
MAX_COMMAND_LENGTH = 120
HELP_TEXT = "let name = expr | expr | :reset | :help"
OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
       "/": lambda a, b: a // b if b else _zero(), "%": lambda a, b: a % b if b else _zero()}


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def _zero():
    raise ReplError("division", "division by zero")


class ReplError(Exception):
    def __init__(self, code, message, position=None):
        super().__init__(message)
        self.code = code
        self.position = position


class Budget:
    def __init__(self, limit):
        self.limit = limit
        self.used = 0

    def spend(self, units=1):
        self.used += units
        if self.used > self.limit:
            raise ReplError("budget", f"step budget {self.limit} exceeded")


class Interpreter:
    def __init__(self, tokens, variables, budget):
        self.tokens, self.variables, self.budget, self.index = tokens, variables, budget, 0

    def peek(self):
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def take(self):
        token = self.peek()
        if token is None:
            raise ReplError("parse", "unexpected end of input", self.index)
        self.index += 1
        return token

    def expression(self):
        value = self.term()
        while self.peek() in ("+", "-"):
            operator = self.take()
            self.budget.spend()
            value = OPS[operator](value, self.term())
        return value

    def term(self):
        value = self.factor()
        while self.peek() in ("*", "/", "%"):
            operator = self.take()
            self.budget.spend()
            value = OPS[operator](value, self.factor())
        return value

    def factor(self):
        token = self.peek()
        if token == "-":
            self.take()
            self.budget.spend()
            return -self.factor()
        if token == "(":
            self.take()
            value = self.expression()
            if self.take() != ")":
                raise ReplError("parse", "missing closing parenthesis", self.index - 1)
            return value
        if token is not None and token.isdigit():
            self.take()
            self.budget.spend()
            return int(token)
        if token is None or not (token[0].isalpha() or token[0] == "_"):
            raise ReplError("parse", f"expected a number or a name, got {token!r}", self.index)
        self.take()
        self.budget.spend()
        if self.peek() == "(":
            return self.call(token)
        if token not in self.variables:
            raise ReplError("unknown-name", f"undefined name {token!r}", self.index - 1)
        return self.variables[token]

    def call(self, name):
        if name != "sum":
            raise ReplError("unknown-function", f"unknown function {name!r}", self.index - 1)
        self.take()                                  # the '(' peeked at by factor
        self.budget.spend()
        arguments = []
        while self.peek() != ")":
            if arguments and self.take() != ",":
                raise ReplError("parse", "expected ',' between arguments", self.index - 1)
            arguments.append(self.expression())
        self.take()
        if len(arguments) != 2:
            raise ReplError("arity", "sum takes exactly two arguments", self.index)
        low, high = arguments
        total = 0
        for _ in range(min(high, low), high + 1):
            self.budget.spend()
            total += 1
        return total

    def run(self):
        value = self.expression()
        if self.peek() is not None:
            raise ReplError("parse", f"trailing token {self.peek()!r}", self.index)
        return value


def execute(command, variables, budget=DEFAULT_BUDGET):
    interpreter = Interpreter(TOKEN_RE.findall(command), dict(variables), Budget(budget))
    return interpreter.run(), interpreter.budget.used


class Session:
    def __init__(self, budget=DEFAULT_BUDGET, max_length=MAX_COMMAND_LENGTH):
        self.variables, self.budget = {}, budget
        self.max_length, self.requests = max_length, 0

    def handle(self, command):
        """Request -> response; the session assigns and echoes the request id."""
        self.requests += 1
        response = {"id": self.requests, "ok": False, "value": None, "steps": 0,
                    "error": None, "error_position": None}
        try:
            if len(command) > self.max_length:
                raise ReplError("limit", f"command longer than {self.max_length} characters")
            value, steps, name = self.run(command.strip())
            response.update({"ok": True, "value": value if name is None else name, "steps": steps})
            if name is not None:
                self.variables[name] = value
        except ReplError as error:
            response.update({"error": f"{error.code}: {error}", "error_position": error.position})
        return response

    def run(self, command):
        """Returns (value, steps, binding); raises ReplError for anything the client got wrong."""
        if not command:
            raise ReplError("parse", "empty command", 0)
        if command == ":reset":
            self.variables.clear()
            return None, 0, None
        if command == ":help":
            return HELP_TEXT, 0, None
        if command.startswith(":"):
            raise ReplError("unknown-command", f"unknown command {command.split()[0]!r}")
        if command.startswith("let "):
            head, _, tail = command[4:].partition("=")
            name = head.strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name or "") or not tail.strip():
                raise ReplError("parse", "let needs a name and an expression", 0)
            value, steps = execute(tail, self.variables, self.budget)
            return value, steps, name
        value, steps = execute(command, self.variables, self.budget)
        return value, steps, None


def main():
    session = Session()
    script = ["2 + 3 * 4", "let base = 7", "base * base - 1", "(2 + 3) * (4 - 1)", "- 4 % 3"]
    responses = [session.handle(command) for command in script]
    # 1) happy path and response shape: every command answers, the binding persists, ids are
    #    assigned in order, and each row is JSON-shaped and self-consistent.
    check([row["ok"] for row in responses] == [True] * 5
          and [row["value"] for row in responses] == [14, "base", 48, 15, 2]
          and [row["id"] for row in responses] == [1, 2, 3, 4, 5]
          and session.variables == {"base": 7}
          and sorted(responses[0]) == ["error", "error_position", "id", "ok", "steps", "value"]
          and all((row["error"] is None and row["value"] is not None) if row["ok"]
                  else (row["error"] is not None and row["value"] is None)
                  for row in responses),
          f"the REPL script went wrong: {[(r['value'], r['error']) for r in responses]}")
    # 2) oracle: Python evaluates the shared integer subset (`/` is this sandbox's floor division).
    literals = ["1 + 2 * 3", "(1 + 2) * 3", "100 % 7", "20 - 5 - 5", "((2))", "7 % 4"]
    ours = [session.handle(command)["value"] for command in literals]
    theirs = [eval(command) for command in literals]
    check(ours == theirs == [7, 9, 2, 10, 2, 3] and session.handle("100 / 7")["value"] == 14,
          f"the interpreter disagrees with the oracle: {ours} vs {theirs}")
    # 3) step accounting: `sum` charges the name, the call, both literals and one step per
    #    iteration; a flat expression costs one step per arithmetic token.
    loop = session.handle("let total = sum(1, 50)")
    counted = [Session().handle(command)["steps"] for command in
               ["1", "1 + 2", "8 % 3 + 4", "10 - 2 * 3", "1 + 2 + 3"]]
    check(loop["ok"] and session.variables["total"] == 50 and loop["steps"] == 50 + 2 + 2
          and counted == [1, 3, 5, 5, 5],
          f"step accounting wrong: {loop['steps']} / {counted}")
    # 4) sandbox budget and atomicity: a runaway loop is cut off, the session survives, and no
    #    failed request ever leaves a partial binding behind.
    runaway = session.handle("let huge = sum(1, 100000)")
    before = dict(session.variables)
    failures = [session.handle("let broken = 1 / 0"), session.handle("let ghost = missing + 1"),
                session.handle("let weird = sum(1)"), session.handle("3 +")]
    check(not runaway["ok"] and runaway["error"].startswith("budget:")
          and runaway["value"] is None and "huge" not in session.variables
          and session.handle("1 + 1")["ok"]
          and [row["error"].split(":")[0] for row in failures]
          == ["division", "unknown-name", "arity", "parse"]
          and session.variables == before and all(row["value"] is None for row in failures),
          f"budget/atomicity wrong: {runaway['error']} / {[row['error'] for row in failures]}")
    # 5) protocol and boundaries: parse errors carry a token position, ids keep increasing, empty
    #    and over-long commands are refused, :reset clears state, :help answers.
    positioned = session.handle("2 * - )")
    reset = session.handle(":reset")
    check(positioned["id"] == reset["id"] - 1 and positioned["error_position"] == 3
          and positioned["error"].startswith("parse:")
          and session.handle(":nope")["error"].startswith("unknown-command:")
          and session.handle("2 * - )")["id"] == session.requests
          and not session.handle("")["ok"] and session.handle("")["error"].startswith("parse:")
          and session.handle("1 + " + "1" * 200)["error"].startswith("limit:")
          and reset["ok"] and session.variables == {}
          and session.handle(":help")["value"] == HELP_TEXT
          and not session.handle("base")["ok"],
          f"protocol/boundaries wrong: {positioned}")
    # 6) determinism and isolation: a fresh session replaying the script answers identically (ids
    #    included) and shares no variables.
    replay = Session()
    check([replay.handle(command) for command in script] == responses
          and replay.variables == {"base": 7} and session.variables == {},
          "replaying the script must reproduce the responses exactly")
    print(f"values={[row['value'] for row in responses]} flat_steps={counted} oracle={theirs}")
    print("PASS")


if __name__ == "__main__":
    main()

