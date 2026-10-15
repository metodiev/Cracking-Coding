"""Task 191 - Convert infix to postfix (shunting-yard simplified). (Tier: Beginner, CTCI Ch 3)

Shunting-yard conversion for + - * / and parentheses. Tokens are whitespace-separated
and the output is a space-separated postfix token string. All operators are
left-associative; unary operators and functions are not supported. Mismatched
parentheses or unknown tokens raise ValueError.
"""

PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2}


def infix_to_postfix(expr):
    tokens = expr.split()
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
        else:
            raise ValueError("unknown token: %r" % token)
    while stack:
        if stack[-1] == "(":
            raise ValueError("unmatched '(' in expression")
        output.append(stack.pop())
    return " ".join(output)


def main():
    assert infix_to_postfix("3 + 4 * 2") == "3 4 2 * +"
    assert infix_to_postfix("( 1 + 2 ) * 3") == "1 2 + 3 *"
    assert infix_to_postfix("1 + 2 - 3") == "1 2 + 3 -"          # left-associative
    assert infix_to_postfix("8 / 4 / 2") == "8 4 / 2 /"          # left-associative
    assert infix_to_postfix("2 * 3 + 4 * 5") == "2 3 * 4 5 * +"
    assert infix_to_postfix("( 1 + 2 ) * ( 3 - 4 )") == "1 2 + 3 4 - *"
    assert infix_to_postfix("( ( 12 ) )") == "12"
    assert infix_to_postfix("") == ""

    for bad in ("1 + 2 )", "( 1 + 2", "( ( 1 ) + 2 ) )", "1 + a", "1 ^ 2"):
        try:
            infix_to_postfix(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected error for %r" % bad)


if __name__ == "__main__":
    main()
    print("PASS")
