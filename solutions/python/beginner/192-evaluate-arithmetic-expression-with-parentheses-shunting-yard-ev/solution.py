"""Task 192 - Evaluate arithmetic expression with +,-,*,/, parentheses (shunting-yard + eval). (Tier: Beginner, CTCI Ch 3)

Pipeline: tokenize -> infix-to-postfix (shunting-yard, left-associative + - * /) -> evaluate
with a stack. Integer arithmetic only; division truncates toward zero (like C/Java:
( 0 - 7 ) / 2 == -3). Malformed expressions and division by zero raise ValueError.
"""

PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2}


def tokenize(expr):
    tokens = expr.split()
    for token in tokens:
        if not (token.isdigit() or token in PRECEDENCE or token in "()"):
            raise ValueError("unknown token: %r" % token)
    return tokens


def infix_to_postfix(tokens):
    output = []
    stack = []
    for token in tokens:
        if token.isdigit():
            output.append(token)
        elif token in PRECEDENCE:
            while stack and stack[-1] in PRECEDENCE and PRECEDENCE[stack[-1]] >= PRECEDENCE[token]:
                output.append(stack.pop())
            stack.append(token)
        elif token == "(":
            stack.append(token)
        elif token == ")":
            while stack and stack[-1] != "(":
                output.append(stack.pop())
            if not stack:
                raise ValueError("unmatched ')' in expression")
            stack.pop()
    while stack:
        if stack[-1] == "(":
            raise ValueError("unmatched '(' in expression")
        output.append(stack.pop())
    return output


def evaluate_postfix(tokens):
    stack = []
    for token in tokens:
        if token in PRECEDENCE:
            if len(stack) < 2:
                raise ValueError("malformed expression: operator without operands")
            right = stack.pop()
            left = stack.pop()
            stack.append(_apply(token, left, right))
        else:
            stack.append(int(token))
    if len(stack) != 1:
        raise ValueError("malformed expression: stray operands")
    return stack[0]


def _apply(op, left, right):
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if right == 0:
        raise ValueError("division by zero")
    quotient = abs(left) // abs(right)              # truncate toward zero
    return quotient if (left >= 0) == (right >= 0) else -quotient


def evaluate(expr):
    return evaluate_postfix(infix_to_postfix(tokenize(expr)))


def main():
    assert evaluate("3 + 4 * 2") == 11
    assert evaluate("( 1 + 2 ) * 3") == 9
    assert evaluate("10 / 3") == 3
    assert evaluate("2 * ( 3 + 4 ) - 5") == 9
    assert evaluate("( 0 - 7 ) / 2") == -3          # truncation toward zero, not floor
    assert evaluate("100 / 7 / 2") == 7             # ((100 / 7) = 14) / 2 = 7
    assert evaluate("7") == 7
    assert evaluate("( ( 2 + 3 ) * ( 4 - 1 ) )") == 15

    for bad in ("10 / 0", "( 1 + 2", "1 + 2 )", "1 +", "", "3 @ 4", "( 1 + 2 ) )"):
        try:
            evaluate(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected error for %r" % bad)


if __name__ == "__main__":
    main()
    print("PASS")
