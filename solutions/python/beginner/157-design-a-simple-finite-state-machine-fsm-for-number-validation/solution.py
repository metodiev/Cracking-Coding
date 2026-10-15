"""Task 157 - Design a simple FSM for number validation. (Tier: Beginner, CTCI Ch 7)

A data-driven finite state machine that validates signed decimal literals.
Exponent notation (e.g. "1e5") is intentionally NOT supported: the FSM has no
'e' transition and such strings are rejected.

States: START -> SIGN -> INT -> DOT -> FRAC
    * digits move START/SIGN -> INT and INT/DOT/FRAC -> FRAC
    * '+'/'-' are only allowed at START
    * '.' is only allowed after at least one digit, so ".5" is invalid
    * a valid number must end in an accepting state (INT or FRAC), so "1.",
      "+" and "" are invalid

_TRANSITIONS maps (state, character class) -> next state; accepts() simply
runs every character through the table and checks the final state, so the
machine is data-driven rather than a pile of if-statements.
"""

START, SIGN, INT, DOT, FRAC = "START", "SIGN", "INT", "DOT", "FRAC"
ACCEPTING = frozenset({INT, FRAC})

_TRANSITIONS = {
    (START, "digit"): INT,
    (START, "sign"): SIGN,
    (SIGN, "digit"): INT,
    (INT, "digit"): INT,
    (INT, "dot"): DOT,
    (DOT, "digit"): FRAC,
    (FRAC, "digit"): FRAC,
}


def _char_class(ch):
    if "0" <= ch <= "9":
        return "digit"
    if ch in "+-":
        return "sign"
    if ch == ".":
        return "dot"
    return None


def accepts(s):
    """True iff `s` is a signed decimal number accepted by the FSM above."""
    state = START
    for ch in s:
        cls = _char_class(ch)
        if cls is None:
            return False
        state = _TRANSITIONS.get((state, cls))
        if state is None:
            return False
    return state in ACCEPTING


def main():
    assert accepts("123") is True
    assert accepts("+1") is True
    assert accepts("-1.5") is True
    assert accepts("0") is True
    assert accepts("-0.0") is True
    assert accepts("007") is True                 # leading zeros are fine
    assert accepts("1.") is False                 # DOT is not an accepting state
    assert accepts(".5") is False                 # a leading digit is required
    assert accepts("") is False                   # must end in INT or FRAC
    assert accepts("+") is False
    assert accepts("-") is False
    assert accepts(".") is False
    assert accepts("1.2.3") is False
    assert accepts("12a") is False
    assert accepts(" 12") is False                # whitespace is invalid everywhere
    assert accepts("12 ") is False
    assert accepts("+-1") is False
    assert accepts("1e5") is False                # exponents are intentionally unsupported

    # the transition table is explicit and data-driven
    assert (START, "dot") not in _TRANSITIONS         # ".5" cannot start with a dot
    assert _TRANSITIONS[(INT, "dot")] == DOT
    assert _TRANSITIONS[(DOT, "digit")] == FRAC
    assert ACCEPTING == {INT, FRAC}


if __name__ == "__main__":
    main()
    print("PASS")
