"""Task 193 - Implement simple expression tokenizer. (Tier: Beginner, CTCI Ch 3)

Converts an expression string into a token stream: INT(n) for integers, OP(op) for
+ - * /, LPAREN / RPAREN for parentheses. Whitespace is ignored anywhere in the input.
Unknown characters raise ValueError with the character and its index; empty (or blank)
input yields an empty token list.
"""

OPERATORS = "+-*/"


class Token:
    def __init__(self, type_, value):
        self.type = type_
        self.value = value

    def __eq__(self, other):
        return isinstance(other, Token) and self.type == other.type and self.value == other.value

    def __repr__(self):
        return "Token(%s, %r)" % (self.type, self.value)


def tokenize(expr):
    tokens = []
    index = 0
    while index < len(expr):
        ch = expr[index]
        if ch.isspace():
            index += 1
        elif ch.isdigit():
            end = index
            while end < len(expr) and expr[end].isdigit():
                end += 1
            tokens.append(Token("INT", int(expr[index:end])))
            index = end
        elif ch in OPERATORS:
            tokens.append(Token("OP", ch))
            index += 1
        elif ch == "(":
            tokens.append(Token("LPAREN", ch))
            index += 1
        elif ch == ")":
            tokens.append(Token("RPAREN", ch))
            index += 1
        else:
            raise ValueError("unknown character %r at index %d" % (ch, index))
    return tokens


def main():
    tokens = tokenize("12 + (3*4)")
    assert [t.type for t in tokens] == ["INT", "OP", "LPAREN", "INT", "OP", "INT", "RPAREN"]
    assert [t.value for t in tokens] == [12, "+", "(", 3, "*", 4, ")"]
    assert tokens == [
        Token("INT", 12), Token("OP", "+"), Token("LPAREN", "("),
        Token("INT", 3), Token("OP", "*"), Token("INT", 4), Token("RPAREN", ")"),
    ]

    assert tokenize("1+2") == [Token("INT", 1), Token("OP", "+"), Token("INT", 2)]
    assert tokenize("  7  ") == [Token("INT", 7)]
    assert tokenize("") == []
    assert tokenize(" \t\n ") == []
    assert tokenize("10/2-3") == [
        Token("INT", 10), Token("OP", "/"), Token("INT", 2), Token("OP", "-"), Token("INT", 3),
    ]

    try:
        tokenize("12.5")
    except ValueError as exc:
        assert "'.'" in str(exc) and "index 2" in str(exc), str(exc)
    else:
        raise AssertionError("expected error for '12.5'")

    for bad in ("a+b", "1 $ 2", "#"):
        try:
            tokenize(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected error for %r" % bad)


if __name__ == "__main__":
    main()
    print("PASS")
