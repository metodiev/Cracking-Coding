"""Task 751 - Implement grammar-based fuzzing for structured inputs.  (Tier: Expert, Ch 11 - Testing)

A grammar-based fuzzer derives inputs *from a grammar* instead of throwing random bytes at the
program, so every generated input is syntactically valid by construction.  The grammar here
describes a tiny arithmetic language:

    <start>  -> <expr>
    <expr>   -> <term> ( <addop> <term> ){0,3}
    <term>   -> <factor> ( <mulop> <factor> ){0,2}
    <factor> -> <number> | '(' <expr> ')' | '-' <factor>
    <number> -> <digit> <digit>{0,2}          <digit> -> '0'..'9'
    <addop>  -> '+' | '-'                     <mulop> -> '*' | '/'

Each rule is expanded with a *depth budget*.  `rule_depths` computes the least fixpoint of the
minimum derivation depth of every rule (1 per rule expansion, plus the depth of the referenced
rules); an alternative is only eligible when it fits the remaining budget, which is what makes
the recursive `<factor> -> '-' <factor>` production terminate instead of looping forever, and a
purely cyclic grammar receives depth infinity (checked below).  Which alternative is taken and
how often a repetition is unrolled come from a hand-written splitmix64 PRNG, so runs are exactly
reproducible.

The fuzzer runs in three modes: `generate` (a fresh input per iteration - 100% valid by
construction), `mutate` (coverage-guided mutation of a corpus of valid inputs, where coverage is
the set of AST features seen so far) and `random` (uniform noise over an alphabet that contains
the language characters *and* junk).  The target is `evaluate`, whose deliberate defect is that
it does not guard the division: dividing by an expression that evaluates to zero raises
ZeroDivisionError - the "crash" the fuzzer hunts.  A found crash is then *shrunk* by deleting
single characters and canonicalising digits to '0' until no further reduction still crashes.

Simplified versus a production fuzzer (e.g. AFL/libFuzzer + Nautilus): inputs are strings, not
bytes; coverage is a coarse AST-feature set rather than edge coverage of instrumented code; the
mutation operators are the four classic ones; there is no fork server or time-out handling.
Invariants that are still verified: every generated input parses with an independent
recursive-descent parser, the generation depth is respected, the grammar's minimum derivation
depth is respected (a too-small budget is refused), the fuzzer really finds the planted defect
while byte noise does not, and the shrunk crash is a local minimum that still crashes, still
parses, and provably contains a division whose divisor evaluates to zero.
"""
MASK64 = (1 << 64) - 1
INF = 1 << 30
MAX_DEPTH = 9
LENGTH_CAP = 1000


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


# --------------------------------------------------------------------------- grammar model

def lit(text):
    return ("lit", text)


def ref(name):
    return ("ref", name)


def rep(items, lo, hi):
    return ("rep", tuple(items), lo, hi)


GRAMMAR = {
    "<start>": ((ref("<expr>"),),),
    "<expr>": ((ref("<term>"), rep([ref("<addop>"), ref("<term>")], 0, 3)),),
    "<term>": ((ref("<factor>"), rep([ref("<mulop>"), ref("<factor>")], 0, 2)),),
    "<factor>": ((ref("<number>"),),
                 (lit("("), ref("<expr>"), lit(")")),
                 (lit("-"), ref("<factor>"))),
    "<number>": ((ref("<digit>"), rep([ref("<digit>")], 0, 2)),),
    "<digit>": tuple((lit(d),) for d in "0123456789"),
    "<addop>": ((lit("+"),), (lit("-"),)),
    "<mulop>": ((lit("*"),), (lit("/"),)),
}
START = "<start>"


def item_depth(item, depths):
    """Minimum depth a single grammar item still needs, given the per-rule depth table."""
    if item[0] == "lit":
        return 0
    if item[0] == "ref":
        return depths.get(item[1], INF)
    sub, lo = item[1], item[2]
    if lo == 0:                       # an optional repetition may be skipped entirely
        return 0
    return max((item_depth(i, depths) for i in sub), default=0)


def alt_depth(alt, depths):
    """A rule expansion costs one depth unit plus the deepest item it references."""
    return 1 + max((item_depth(i, depths) for i in alt), default=-1)


def rule_depths(rules):
    """Least fixpoint of per-rule minimum derivation depth; INF means no finite derivation."""
    depths = {name: INF for name in rules}
    changed = True
    while changed:
        changed = False
        for name, alternatives in rules.items():
            best = min(alt_depth(alt, depths) for alt in alternatives)
            if best < depths[name]:
                depths[name] = best
                changed = True
    return depths


class Generator:
    """Derives strings from a grammar under a depth budget, deterministically."""

    def __init__(self, rules, depth):
        self.rules = rules
        self.depths = rule_depths(rules)
        self.depth = depth
        self.productions_used = set()

    def _items(self, items, depth, state):
        chunks = []
        for item in items:
            if item[0] == "lit":
                chunks.append(item[1])
            elif item[0] == "ref":
                text, state = self._rule(item[1], depth - 1, state)
                chunks.append(text)
            else:
                sub, lo, hi = item[1], item[2], item[3]
                state, r = splitmix64(state)
                inner = max((item_depth(i, self.depths) for i in sub), default=0)
                extra = r % (hi - lo + 1)
                # a repeated group is only unrolled when its items still fit the budget: the
                # group's items are expanded with depth-1, and their own references need
                # another level, hence inner + 2 <= depth
                count = lo + (extra if inner + 2 <= depth else 0)
                for _ in range(count):
                    text, state = self._items(sub, depth - 1, state)
                    chunks.append(text)
        return "".join(chunks), state

    def _rule(self, name, depth, state):
        alternatives = self.rules[name]
        options = [i for i, alt in enumerate(alternatives)
                   if alt_depth(alt, self.depths) <= depth]
        check(options, "no alternative of %s fits the remaining depth %d" % (name, depth))
        state, r = splitmix64(state)
        index = options[r % len(options)]
        self.productions_used.add((name, index))
        return self._items(alternatives[index], depth, state)

    def generate(self, state):
        return self._rule(START, self.depth, state)


# --------------------------------------------------------------------- parser / evaluator

LANGUAGE_CHARS = set("0123456789+-*/()")
JUNK_ALPHABET = "0123456789+-*/(){}[];:,.abcXYZ \t\n@#$%"


def tokenize(text):
    tokens = []
    for ch in text:
        if ch not in LANGUAGE_CHARS:
            raise SyntaxError("illegal character %r" % ch)
        tokens.append(ch)
    return tokens


class Parser:
    """Independent recursive-descent parser: the oracle the generator is never allowed to use."""

    def __init__(self, text):
        self.tokens = tokenize(text)
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self, expected=None):
        token = self.peek()
        if token is None:
            raise SyntaxError("unexpected end of input")
        if expected is not None and token != expected:
            raise SyntaxError("expected %r, found %r" % (expected, token))
        self.pos += 1
        return token

    def expr(self):
        node = self.term()
        while self.peek() in ("+", "-"):
            op = self.take()
            node = ("bin", op, node, self.term())
        return node

    def term(self):
        node = self.factor()
        while self.peek() in ("*", "/"):
            op = self.take()
            node = ("bin", op, node, self.factor())
        return node

    def factor(self):
        token = self.peek()
        if token == "(":
            self.take("(")
            node = self.expr()
            self.take(")")
            return node
        if token == "-":
            self.take("-")
            return ("neg", self.factor())
        if token is not None and token.isdigit():
            digits = ""
            while self.peek() is not None and self.peek().isdigit():
                digits += self.take()
            return ("num", int(digits))
        raise SyntaxError("expected a factor, found %r" % token)


def parse(text):
    parser = Parser(text)
    node = parser.expr()
    if parser.pos != len(parser.tokens):
        raise SyntaxError("trailing input at token %d" % parser.pos)
    return node


def evaluate(node):
    if node[0] == "num":
        return node[1]
    if node[0] == "neg":
        return -evaluate(node[1])
    op = node[1]
    left = evaluate(node[2])
    right = evaluate(node[3])
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if right == 0:                       # the planted defect of the target program
        raise ZeroDivisionError("division by zero in %r" % (op,))
    return left // right


def tree_depth(node):
    if node[0] == "num":
        return 0
    if node[0] == "neg":
        return 1 + tree_depth(node[1])
    return 1 + max(tree_depth(node[2]), tree_depth(node[3]))


def zero_divisors(node, found):
    """Division nodes whose right operand evaluates to zero: the semantic cause of the crash."""
    if node[0] == "neg":
        zero_divisors(node[1], found)
    elif node[0] == "bin":
        if node[1] == "/":
            try:
                divisor = evaluate(node[3])
            except ZeroDivisionError:
                divisor = None
            if divisor == 0:
                found.append(node)
        zero_divisors(node[2], found)
        zero_divisors(node[3], found)
    return found


def collect_numbers(node, into):
    if node[0] == "num":
        into.append(node[1])
    elif node[0] == "neg":
        collect_numbers(node[1], into)
    else:
        collect_numbers(node[2], into)
        collect_numbers(node[3], into)
    return into


def ast_features(node, into):
    if node[0] == "num":
        into.add("num")
    elif node[0] == "neg":
        into.add("neg")
        ast_features(node[1], into)
    else:
        into.add("bin" + node[1])
        ast_features(node[2], into)
        ast_features(node[3], into)
    into.add("depth%d" % tree_depth(node))
    return into


def crashes(text):
    try:
        tree = parse(text)
    except SyntaxError:
        return False
    try:
        evaluate(tree)
    except ZeroDivisionError:
        return True
    return False


# ------------------------------------------------------------------------------- fuzzer

def rand_string(state, length):
    out = []
    for _ in range(length):
        state, r = splitmix64(state)
        out.append(JUNK_ALPHABET[r % len(JUNK_ALPHABET)])
    return "".join(out), state


def mutate(text, state, alphabet="0123456789+-*/()"):
    if not text:
        text = "0"
    state, r = splitmix64(state)
    op = r % 4
    state, r = splitmix64(state)
    index = r % len(text)
    state, r = splitmix64(state)
    char = alphabet[r % len(alphabet)]
    if op == 0 and len(text) > 1:
        return text[:index] + text[index + 1:], state
    if op == 1:
        return text[:index] + char + text[index:], state
    if op == 2:
        return text[:index] + char + text[index + 1:], state
    return text[:index] + text[index] + text[index:], state


def fuzz(iterations, seed, mode, generator):
    state = seed & MASK64
    corpus = []
    if mode in ("generate", "mutate"):
        for _ in range(8):
            text, state = generator.generate(state)
            corpus.append(text)
    covered = set()
    parse_ok = rejected = generated = 0
    crashes_found = []
    for _ in range(iterations):
        if mode == "generate":
            text, state = generator.generate(state)
            generated += 1
        elif mode == "mutate":
            state, r = splitmix64(state)
            parent = corpus[r % len(corpus)]
            text, state = mutate(parent, state)
        else:
            state, r = splitmix64(state)
            text, state = rand_string(state, 20 + r % 21)
            generated += 1
        try:
            tree = parse(text)
        except SyntaxError:
            rejected += 1
            continue
        parse_ok += 1
        try:
            evaluate(tree)
        except ZeroDivisionError:
            crashes_found.append(text)
            continue
        features = ast_features(tree, set())
        fresh = features - covered
        if fresh:
            covered |= fresh
            corpus.append(text)
    return {"corpus": corpus, "covered": covered, "parse_ok": parse_ok,
            "rejected": rejected, "crashes": crashes_found, "generated": generated}


def shrink(text):
    """Greedy delta debugging: drop single characters, then canonicalise digits to '0'."""
    current = text
    improved = True
    while improved:
        improved = False
        for i in range(len(current)):
            trial = current[:i] + current[i + 1:]
            if trial and crashes(trial):
                current, improved = trial, True
                break
        if improved:
            continue
        for i, ch in enumerate(current):
            if ch.isdigit() and ch != "0" and crashes(current[:i] + "0" + current[i + 1:]):
                current, improved = current[:i] + "0" + current[i + 1:], True
                break
    return current


def main():
    depths = rule_depths(GRAMMAR)

    # 1. the depth fixpoint: minimal derivations and a grammar that can never terminate
    check(depths["<digit>"] == 1 and depths["<addop>"] == 1, "terminal rules derive in one step")
    check(depths["<number>"] == 2 and depths["<factor>"] == 3, "derivation depths of <number>/<factor>")
    check(depths[START] == 6, "the shortest derivation of <start> is 6 rule expansions deep")
    cyclic = {"<a>": ((ref("<b>"),),), "<b>": ((ref("<a>"),),)}
    check(rule_depths(cyclic)["<a>"] >= INF, "a cyclic grammar must have no finite derivation depth")

    # 2. everything the generator emits parses with the independent parser, inside the budget
    generator = Generator(GRAMMAR, MAX_DEPTH)
    state = 0x5EED
    produced = []
    for _ in range(200):
        text, state = generator.generate(state)
        produced.append(text)
        check(len(text) <= LENGTH_CAP, "generated input must stay small: %d chars" % len(text))
        tree = parse(text)                                   # raises if the generator is wrong
        check(tree_depth(tree) <= MAX_DEPTH, "parse tree must respect the depth budget")
    check(len(set(produced)) > 150, "the generator must keep producing new inputs")

    # 3. production coverage: every rule, and nearly every alternative, is exercised
    rules_used = {name for name, _ in generator.productions_used}
    check(rules_used == set(GRAMMAR), "every rule must be covered, missing: %s"
          % (set(GRAMMAR) - rules_used))
    total_alts = sum(len(alts) for alts in GRAMMAR.values())
    check(len(generator.productions_used) >= 0.9 * total_alts,
          "production coverage must reach ~90%%: %d/%d"
          % (len(generator.productions_used), total_alts))

    # 4. determinism: same seed, same inputs
    twin = Generator(GRAMMAR, MAX_DEPTH)
    state_b, produced_b = 0x5EED, []
    for _ in range(50):
        text, state_b = twin.generate(state_b)
        produced_b.append(text)
    check(produced_b == produced[:50], "generation must be deterministic")
    check(twin.productions_used <= generator.productions_used,
          "the same seed prefix must use the same productions")

    # 5. the depth budget is enforced, and the minimal budget only affords the cheap alternatives
    try:
        Generator(GRAMMAR, depths[START] - 1).generate(1)
        raise AssertionError("a too small depth budget must be refused")
    except AssertionError as exc:
        check("fits the remaining depth" in str(exc), "refusal must name the reason")
    minimal, _ = Generator(GRAMMAR, depths[START]).generate(7)
    check(len(minimal) == 1 and minimal.isdigit(),
          "the minimal budget must yield a one-digit expression, got %r" % minimal)
    minimal_tree = parse(minimal)
    numbers = collect_numbers(minimal_tree, [])
    check(numbers and all(0 <= value <= 9 for value in numbers),
          "at the minimal budget every number is a single digit, got %r" % minimal)
    check("(" not in minimal and "neg" not in ast_features(minimal_tree, set()),
          "at the minimal budget neither parentheses nor unary minus are affordable: %r" % minimal)

    # 6. grammar-derived fuzzing: always valid, and it finds the planted division-by-zero defect
    grammar_run = fuzz(300, 0xC0FFEE, "generate", Generator(GRAMMAR, MAX_DEPTH))
    check(grammar_run["rejected"] == 0 and grammar_run["parse_ok"] == 300,
          "generated inputs must all parse (%d rejected)" % grammar_run["rejected"])
    check(grammar_run["crashes"], "grammar fuzzing must reach the division-by-zero defect")

    # 7. coverage-guided mutation of a valid corpus: still finds crashes, many mutations rejected
    mutation_run = fuzz(2000, 0xBEEF, "mutate", Generator(GRAMMAR, MAX_DEPTH))
    check(mutation_run["crashes"], "mutation-based fuzzing must also find the defect")
    check(mutation_run["rejected"] > 0, "byte mutations must destroy syntax often enough to matter")
    check(len(mutation_run["covered"]) >= len(grammar_run["covered"]),
          "the mutated corpus must not lose feature coverage")

    # 8. uniform noise over the same alphabet: essentially never a valid input, never a crash
    random_run = fuzz(400, 0xDEAD, "random", Generator(GRAMMAR, MAX_DEPTH))
    check(random_run["parse_ok"] == 0 and not random_run["crashes"],
          "uniform noise must not produce valid inputs (%d parsed, %d crashed)"
          % (random_run["parse_ok"], len(random_run["crashes"])))

    # 9. shrinking: smaller, still crashing, still valid, and locally minimal
    original = grammar_run["crashes"][0]
    small = shrink(original)
    check(len(small) <= len(original), "shrinking must not grow the input")
    check(crashes(small), "the shrunk input must still crash")
    for i in range(len(small)):
        check(not crashes(small[:i] + small[i + 1:]), "no single deletion may still crash: %r" % small)
    for i, ch in enumerate(small):
        if ch.isdigit() and ch != "0":
            check(not crashes(small[:i] + "0" + small[i + 1:]),
                  "no digit may be canonicalised further: %r" % small)
    causes = zero_divisors(parse(small), [])
    check(causes, "the shrunk crash must have a division by a zero-valued divisor")
    check(len(small) <= 15, "shrinking must collapse a random crash to a tiny witness: %r" % small)
    check(crashes(original) and original != small,
          "the original crash must stay reproducible while a smaller witness exists")

    # 10. rejected inputs raise SyntaxError instead of crashing, and boundaries behave
    for bad in ("", "1+", "(1+2", "1 2", "()", "+1", "1++2"):
        try:
            parse(bad)
            raise AssertionError("invalid input must be rejected: %r" % bad)
        except SyntaxError:
            pass
        check(not crashes(bad), "an unparsable input is not a crash")
    check(evaluate(parse("(1+2)*3")) == 9, "parenthesised arithmetic")
    check(evaluate(parse("-(2+3)")) == -5, "unary minus")
    check(evaluate(parse("8/4")) == 2, "integer division")
    try:
        evaluate(parse("1/0"))
        raise AssertionError("division by zero must raise ZeroDivisionError")
    except ZeroDivisionError:
        pass

    # 11. the whole pipeline is reproducible: identical seed -> identical crash and minimisation
    rerun = fuzz(300, 0xC0FFEE, "generate", Generator(GRAMMAR, MAX_DEPTH))
    check(rerun["crashes"] == grammar_run["crashes"], "fuzzing must be reproducible")
    check(shrink(rerun["crashes"][0]) == small, "minimisation must be reproducible")

    print("751 generate: %d inputs, %d parse failures, %d crashes, %d productions covered"
          % (grammar_run["generated"], grammar_run["rejected"],
             len(grammar_run["crashes"]), len(generator.productions_used)))
    print("751 mutate: %d/%d parsed, %d crashes after %d mutation rounds"
          % (mutation_run["parse_ok"], mutation_run["parse_ok"] + mutation_run["rejected"],
             len(mutation_run["crashes"]), 2000))
    print("751 random: %d/%d parsed, %d crashes (noise cannot reach the defect)"
          % (random_run["parse_ok"], random_run["generated"], len(random_run["crashes"])))
    print("751 shrink: %r (%d chars) -> %r (%d chars), zero divisor confirmed by AST"
          % (original, len(original), small, len(small)))
    print("PASS")


if __name__ == "__main__":
    main()
