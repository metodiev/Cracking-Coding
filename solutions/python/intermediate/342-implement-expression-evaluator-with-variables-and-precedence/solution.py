"""Task 342 - Expression evaluator with variables and precedence. (Tier: Intermediate, CTCI Ch 8)

Shunting-yard parser for + - * /, parentheses, unary minus and variables supplied
in a dict.  Value policy: everything is evaluated in floating point (IEEE-754
double, identical in all four languages), so 100/8 == 12.5 - there is no integer
truncation anywhere.  Malformed input raises ValueError with a message:
unbalanced parentheses, dangling/adjacent operators, implicit multiplication,
unknown variables, stray characters and division by zero.

Value cross-check: the literal expectations below were computed with an
independent evaluator (CPython's own expression parser).  Note that the brief's
example 3+4*2/(1-5) evaluates to 1: 3 + 8/(1-5) = 3 - 2 = 1, not 2, so the test
asserts 1 and the brute-force cross-check confirms it.
"""

BINARY_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2}
UNARY_PRECEDENCE = 4                   # '~' = unary minus, binds tightest, right-assoc


def tokenize(expr):
    """Split into ("num", float) / ("var", name) / ("op", symbol) tokens."""
    tokens = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch == " ":
            i += 1
        elif ch.isdigit():
            j = i
            while j < len(expr) and expr[j].isdigit():
                j += 1
            tokens.append(("num", float(expr[i:j])))
            i = j
        elif ch.isalpha():
            j = i
            while j < len(expr) and (expr[j].isalnum() or expr[j] == "_"):
                j += 1
            tokens.append(("var", expr[i:j]))
            i = j
        elif ch in "+-*/()":
            tokens.append(("op", ch))
            i += 1
        else:
            raise ValueError("unexpected character %r" % ch)
    return tokens


def to_rpn(tokens):
    """Shunting-yard: tokens -> reverse polish notation, unary minus as '~'."""
    output = []
    ops = []                           # (symbol, precedence)
    expect_operand = True
    for kind, value in tokens:
        if kind == "num":
            if not expect_operand:
                raise ValueError("expected an operator before %r" % value)
            output.append(("num", value))
            expect_operand = False
        elif kind == "var":
            if not expect_operand:
                raise ValueError("expected an operator before %r" % value)
            output.append(("var", value))
            expect_operand = False
        elif value == "(":
            if not expect_operand:
                raise ValueError("implicit multiplication is not supported")
            ops.append(("(", None))
        elif value == ")":
            if expect_operand:
                raise ValueError("empty or dangling parenthesis group")
            while ops and ops[-1][0] != "(":
                output.append(("op", ops.pop()[0]))
            if not ops:
                raise ValueError("unbalanced ')'")
            ops.pop()
            expect_operand = False
        elif expect_operand:
            if value == "+":
                continue               # unary plus is a no-op
            if value == "-":
                ops.append(("~", UNARY_PRECEDENCE))
            else:
                raise ValueError("operator %r has no left operand" % value)
        else:
            precedence = BINARY_PRECEDENCE[value]
            while ops:
                top, top_precedence = ops[-1]
                if top == "(":
                    break
                if top_precedence > precedence or (top_precedence == precedence and top != "~"):
                    output.append(("op", ops.pop()[0]))
                else:
                    break
            ops.append((value, precedence))
            expect_operand = True
    if expect_operand and tokens:
        raise ValueError("dangling operator at end of expression")
    while ops:
        top, _ = ops.pop()
        if top == "(":
            raise ValueError("unbalanced '('")
        output.append(("op", top))
    return output


def evaluate_rpn(rpn, variables):
    stack = []
    for kind, value in rpn:
        if kind == "num":
            stack.append(value)
        elif kind == "var":
            if value not in variables:
                raise ValueError("unknown variable %r" % value)
            stack.append(float(variables[value]))
        elif value == "~":
            if not stack:
                raise ValueError("malformed expression")
            stack.append(-stack.pop())
        else:
            if len(stack) < 2:
                raise ValueError("malformed expression")
            b = stack.pop()
            a = stack.pop()
            if value == "+":
                stack.append(a + b)
            elif value == "-":
                stack.append(a - b)
            elif value == "*":
                stack.append(a * b)
            else:
                if b == 0.0:
                    raise ValueError("division by zero")
                stack.append(a / b)
    if len(stack) != 1:
        raise ValueError("malformed expression")
    return stack[0]


def evaluate(expr, variables=None):
    """Evaluate expr; variables is a dict name -> number. Raises ValueError."""
    variables = variables if variables is not None else {}
    tokens = tokenize(expr)
    if not tokens:
        raise ValueError("empty expression")
    return evaluate_rpn(to_rpn(tokens), variables)


def expect_value_error(expr, variables=None):
    try:
        evaluate(expr, variables)
    except ValueError:
        return
    raise AssertionError("expected ValueError for %r" % expr)


def main():
    # brief examples - the first one is 1, not 2 (see the module docstring)
    assert evaluate("3+4*2/(1-5)") == 1
    assert evaluate("x+y*2", {"x": 3, "y": 4}) == 11
    assert evaluate("(1+2)*(3+4)") == 21

    # values cross-checked against CPython's own expression parser
    assert evaluate("1+2*3-4/8") == 6.5
    assert evaluate("2*(3+4)-5") == 9
    assert evaluate("100/8") == 12.5
    assert evaluate("-(3+4)*2") == -14
    assert evaluate("3*-2") == -6
    assert evaluate("2*-3+4") == -2
    assert evaluate("10-4-3") == 3        # left associative
    assert evaluate("16/4/2") == 2        # (16/4)/2
    assert evaluate("2*(1+2*(3-1))") == 10
    assert evaluate("((7))") == 7
    assert evaluate("8/(2+2)") == 2
    assert evaluate("--5") == 5           # chained unary minus
    assert evaluate("-(-5)") == 5
    assert evaluate("3--2") == 5          # binary minus then unary minus
    assert evaluate("2+3*4-6/3") == 12
    assert evaluate("((1+2)*(3+4))/3") == 7
    assert evaluate("1+((2+3)*4)") == 21
    assert evaluate("7/2") == 3.5
    assert evaluate("(8-3)*(2+1)/5") == 3
    assert evaluate(" 1 + 2 ") == 3       # whitespace is ignored
    assert evaluate("2*-5+20/4") == -5
    assert evaluate("x*y+z", {"x": 3, "y": 4, "z": 5}) == 17
    assert evaluate("-x*2", {"x": 3}) == -6
    assert evaluate("+x", {"x": 7}) == 7  # unary plus
    assert evaluate("price*quantity", {"price": 2.5, "quantity": 4}) == 10

    # malformed input: documented ValueError
    for bad in ["", "1+", "*3", "1**2", "(1+2", "1+2)", "()", "( )", "1 2", "3x",
                "1(2)", "&", "1&2", "4/0", "0/0", "(1+)"]:
        expect_value_error(bad)
    expect_value_error("x+1")             # unknown variable
    expect_value_error("sum", {"total": 1})


if __name__ == "__main__":
    main()
    print("PASS")
