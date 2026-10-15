"""Task 190 - Parse and evaluate postfix (RPN) expression. (Tier: Beginner, CTCI Ch 3)

Tokens are separated by whitespace: integer operands and the operators + - * /.
Evaluation is a stack sweep — push operands, pop two for each operator. '/' is
integer division truncating toward zero (like C/Java), and division by zero,
unknown tokens, missing operands or leftover values all raise ValueError.
"""

import re

OPERATORS = ("+", "-", "*", "/")
INTEGER_TOKEN = re.compile(r"-?\d+")


def tokenize_rpn(expression):
    """Split a whitespace-separated RPN expression into tokens."""
    return expression.split()


def eval_rpn(expression):
    """Evaluate `expression` and return its integer result."""
    stack = []
    for token in tokenize_rpn(expression):
        if token in OPERATORS:
            if len(stack) < 2:
                raise ValueError("not enough operands for %r" % token)
            right = stack.pop()
            left = stack.pop()
            if token == "+":
                stack.append(left + right)
            elif token == "-":
                stack.append(left - right)
            elif token == "*":
                stack.append(left * right)
            else:
                if right == 0:
                    raise ValueError("division by zero")
                quotient = abs(left) // abs(right)
                stack.append(-quotient if (left < 0) != (right < 0) else quotient)
        else:
            if not INTEGER_TOKEN.fullmatch(token):
                raise ValueError("invalid token: %r" % token)
            stack.append(int(token))
    if len(stack) != 1:
        raise ValueError("expression leaves %d values on the stack" % len(stack))
    return stack[0]


def main():
    assert tokenize_rpn("5 1 2 + 4 * + 3 -") == ["5", "1", "2", "+", "4", "*", "+", "3", "-"]
    assert tokenize_rpn("") == []
    assert tokenize_rpn("  7   2 / ") == ["7", "2", "/"]

    assert eval_rpn("3 4 +") == 7
    assert eval_rpn("5 1 2 + 4 * + 3 -") == 14
    assert eval_rpn("7 2 /") == 3
    assert eval_rpn("42") == 42
    assert eval_rpn("  42  ") == 42
    assert eval_rpn("2 3 4 * +") == 14
    assert eval_rpn("1 2 + 3 *") == 9
    assert eval_rpn("10 3 - 4 2 / *") == 14
    assert eval_rpn("5 1 - 1 -") == 3
    assert eval_rpn("0 5 /") == 0

    # integer division truncates toward zero
    assert eval_rpn("-7 2 /") == -3
    assert eval_rpn("7 -2 /") == -3
    assert eval_rpn("-7 -2 /") == 3
    assert eval_rpn("-1 3 /") == 0

    for bad in ("1 0 /", "0 0 /", "3 4", "1 2 + 3", "+", "", "   ", "abc", "1 2 &", "1.5 2 +", "+5 1 +", "3 -"):
        try:
            eval_rpn(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for %r" % bad)


if __name__ == "__main__":
    main()
    print("PASS")
