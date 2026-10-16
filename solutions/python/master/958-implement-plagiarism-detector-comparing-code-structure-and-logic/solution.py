"""Task 958 - Implement plagiarism detector comparing code structure and logic patterns.
(Tier: Master, CTCI Ch 11 - Testing)

Detector model: a submission is normalised (comment stripping, identifier canonicalisation by
first-occurrence order) so renaming a variable cannot hide a copy, then compared on two axes - a
token n-gram Jaccard index for logic and a cosine similarity over token-kind bigrams for structure.
The blended score flags pairs above a threshold and a union-find pass groups them into the clusters a
human reviewer inspects; sources below a minimum token count are "insufficient evidence".

Simplified: the code is short pseudo-Python, the tokeniser is a hand-written regex and no
control-flow graph is built.  Verified: identical/renamed copies, comment invariance, the unrelated
program below the threshold, the exact threshold boundary, metric symmetry and range, cluster
composition and the empty/insufficient-evidence boundary.
"""
import re
KEYWORDS = frozenset("def for while if else elif return in range len enumerate class and or not "
                     "None True False".split())
TOKEN_RE = re.compile(r"\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|[A-Za-z_][A-Za-z_0-9]*|\d+\.?\d*"
                      r"|==|!=|<=|>=|\*\*|[-+*/%<>=!(){}\[\],:.]")
COMMENT_RE = re.compile(r"#.*$|//.*$|/\*.*?\*/", re.M)
LOGIC_WEIGHT, STRUCTURE_WEIGHT = 0.6, 0.4
SHINGLE = 3
THRESHOLD = 0.75
MIN_TOKENS = 8   # below this a source is "insufficient evidence" and is never flagged
CODE_A = """def two_sum(nums, target):
    seen = {}
    for index, value in enumerate(nums):
        if target - value in seen:
            return [seen[target - value], index]
        seen[value] = index
    return []
"""
CODE_A_RENAMED = """# same thing, renamed
def pair_lookup(values, wanted):
    lookup = {}
    for position, number in enumerate(values):
        if wanted - number in lookup:
            return [lookup[wanted - number], position]
        lookup[number] = position
    return []
"""
CODE_A_STYLE = """def two_sum(nums,target):   # trimmed whitespace
    seen={}
    for index,value in enumerate(nums):
        if target-value in seen: return [seen[target-value],index]
        seen[value]=index
    return []"""
CODE_A_REFACTORED = """def pair_sum(values, wanted):
    lookup = {}
    for position in range(len(values)):
        number = values[position]
        if wanted - number in lookup:
            return [lookup[wanted - number], position]
        lookup[number] = position
    return []
"""
CODE_B = """def binary_search(items, wanted):
    low = 0
    high = len(items) - 1
    while low <= high:
        middle = (low + high) // 2
        if items[middle] == wanted:
            return middle
        if items[middle] < wanted:
            low = middle + 1
        else:
            high = middle - 1
    return -1
"""
SOURCES = {"a_original": CODE_A, "a_renamed": CODE_A_RENAMED, "a_style": CODE_A_STYLE,
           "a_refactored": CODE_A_REFACTORED, "b_search": CODE_B, "c_empty": "", "d_empty": ""}


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def tokenize(code):
    """Comments are dropped and literals abstracted; the rest becomes (kind, text) pairs."""
    return [(kind(text), text) for text in TOKEN_RE.findall(COMMENT_RE.sub("", code))]


def kind(text):
    if text[0] in "\"'":
        return "STRING"
    if text in KEYWORDS:
        return "KEYWORD"
    if text[0].isdigit():
        return "NUMBER"
    if text[0].isalpha() or text[0] == "_":
        return "NAME"
    return "OP"


def canonical_names(tokens):
    """Rename every identifier to v1, v2, ... in first-occurrence order."""
    mapping, canonical = {}, []
    for token_kind, text in tokens:
        if token_kind == "NAME":
            mapping.setdefault(text, f"v{len(mapping) + 1}")
            text = mapping[text]
        canonical.append((token_kind, text))
    return canonical


def shingles(tokens, size=SHINGLE):
    words = [text for _, text in tokens]
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[index:index + size]) for index in range(len(words) - size + 1)}


def token_jaccard(left, right):
    """Logic similarity: Jaccard over canonicalised token shingles (renaming-invariant)."""
    first = shingles(canonical_names(tokenize(left)))
    second = shingles(canonical_names(tokenize(right)))
    if not first or not second:
        return 1.0 if not first and not second else 0.0
    return len(first & second) / len(first | second)


def structure_cosine(left, right):
    """Structure similarity: cosine over token-kind bigrams, so identifiers are invisible."""
    def profile(code):
        kinds = [token_kind for token_kind, _ in tokenize(code)]
        counts = {}
        for index in range(len(kinds) - 1):
            key = (kinds[index], kinds[index + 1])
            counts[key] = counts.get(key, 0) + 1
        return counts
    first, second = profile(left), profile(right)
    if not first or not second:
        return 1.0 if not first and not second else 0.0
    keys = set(first) | set(second)
    dot = sum(first.get(key, 0) * second.get(key, 0) for key in keys)
    norm = (sum(value * value for value in first.values()) ** 0.5
            * sum(value * value for value in second.values()) ** 0.5)
    return dot / norm if norm else 0.0


def similarity(left, right):
    return (LOGIC_WEIGHT * token_jaccard(left, right)
            + STRUCTURE_WEIGHT * structure_cosine(left, right))


def pairs(sources):
    """All unordered source pairs, each with its blended score."""
    names = sorted(sources)
    return [(names[i], names[j], similarity(sources[names[i]], sources[names[j]]))
            for i in range(len(names)) for j in range(i + 1, len(names))]


def flagged(sources, threshold=THRESHOLD):
    """Only pairs where both sides carry enough tokens are reported, never empty twins."""
    return [(left, right, score) for left, right, score in pairs(sources)
            if score >= threshold and len(tokenize(sources[left])) >= MIN_TOKENS
            and len(tokenize(sources[right])) >= MIN_TOKENS]


def clusters(sources, threshold=THRESHOLD):
    """Union-find over the flagged pairs: one group per suspected family of submissions."""
    parent = {name: name for name in sources}

    def root(name):
        while parent[name] != name:
            name = parent[name]
        return name

    for left, right, _ in flagged(sources, threshold):
        first, second = root(left), root(right)
        if first != second:
            parent[max(first, second)] = parent[min(first, second)]
    groups = {}
    for name in sorted(sources):
        groups.setdefault(root(name), []).append(name)
    return sorted(groups.values(), key=lambda group: (-len(group), group))


def main():
    # 1) happy path: a copy is identical on both axes (up to the float noise of the cosine).
    check(abs(token_jaccard(CODE_A, CODE_A) - 1.0) < 1e-12
          and abs(structure_cosine(CODE_A, CODE_A) - 1.0) < 1e-12
          and abs(similarity(CODE_A, CODE_A) - 1.0) < 1e-12,
          "a source compared with itself must score 1.0")

    # 2) renaming is invisible after canonicalisation while the raw token streams differ - that
    #    contrast is the oracle for why normalisation is needed.
    raw = " ".join(text for _, text in tokenize(CODE_A))
    raw_renamed = " ".join(text for _, text in tokenize(CODE_A_RENAMED))
    check(abs(token_jaccard(CODE_A, CODE_A_RENAMED) - 1.0) < 1e-12 and raw != raw_renamed
          and similarity(CODE_A, CODE_A_RENAMED) >= 0.95,
          f"a renamed copy must stay detected: {similarity(CODE_A, CODE_A_RENAMED):.3f}")
    # 3) comment and whitespace edits cannot move the score at all.
    check(abs(similarity(CODE_A, CODE_A_STYLE) - 1.0) < 1e-12
          and abs(token_jaccard(CODE_A, CODE_A_STYLE) - 1.0) < 1e-12
          and "# trimmed" not in COMMENT_RE.sub("", CODE_A_STYLE),
          "comments and whitespace must not affect the metric")
    # 4) an unrelated algorithm stays below the threshold and the logic axis is what separates it:
    #    kind-level structure is coarse, so two loop-shaped programs look alike structurally.
    refactored = similarity(CODE_A, CODE_A_REFACTORED)
    unrelated = similarity(CODE_A, CODE_B)
    check(unrelated < THRESHOLD and token_jaccard(CODE_A, CODE_B) < 0.2
          and structure_cosine(CODE_A, CODE_B) > 0.8 and refactored >= THRESHOLD,
          f"unrelated code scored {unrelated:.3f}, refactored copy {refactored:.3f}")

    # 5) metric properties: symmetric, in [0, 1], and inclusive at the exact threshold.
    symmetric = all(abs(similarity(one, two) - similarity(two, one)) < 1e-12
                    and -1e-12 <= similarity(one, two) <= 1.0 + 1e-12
                    for one in SOURCES.values() for two in SOURCES.values())
    check(symmetric and flagged({"a": CODE_A, "b": CODE_A_REFACTORED}, refactored)
          and not flagged({"a": CODE_A, "b": CODE_A_REFACTORED}, refactored + 1e-9),
          f"the metric must be symmetric and inclusive at the threshold ({refactored:.6f})")

    # 6) four versions of one program collapse into a single group; the binary search and the two
    #    empty sources stay singletons.
    groups = clusters(SOURCES)
    check(groups[0] == ["a_original", "a_refactored", "a_renamed", "a_style"]
          and [group for group in groups if len(group) == 1] == [["b_search"], ["c_empty"],
                                                                ["d_empty"]]
          and len(flagged(SOURCES)) == 6,
          f"clustering wrong: {groups}")

    # 7) boundaries: empty sources are identical to each other and unrelated to any program, and a
    #    pair without enough tokens is never reported however similar it looks.
    check(abs(similarity("", "") - 1.0) < 1e-12 and similarity("", CODE_A) == 0.0
          and abs(token_jaccard("# just a comment\n", "\n\n") - 1.0) < 1e-12
          and not flagged({"x": "", "y": ""}) and not flagged({"x": "", "y": CODE_A}),
          "the empty-program boundary must be well defined")
    # 8) determinism and insertion-order independence, including the reported pair count.
    reversed_sources = {name: SOURCES[name] for name in reversed(sorted(SOURCES))}
    check(flagged(SOURCES) == flagged(reversed_sources) and clusters(SOURCES) ==
          clusters(reversed_sources) and len(pairs(SOURCES)) == 21
          and flagged(SOURCES) == sorted(flagged(SOURCES)),
          "detection must be deterministic and order independent")
    # 9) the reported score is the documented weighted blend of the two axes, recomputed by hand.
    manual = [LOGIC_WEIGHT * token_jaccard(CODE_A, code) + STRUCTURE_WEIGHT
              * structure_cosine(CODE_A, code) for code in (CODE_A_RENAMED, CODE_B, CODE_A_STYLE)]
    check(all(abs(value - similarity(CODE_A, code)) < 1e-9
              for value, code in zip(manual, (CODE_A_RENAMED, CODE_B, CODE_A_STYLE))),
          f"the blend formula must match the implementation: {manual}")

    print(f"sources={len(SOURCES)} pairs={len(pairs(SOURCES))} flagged={len(flagged(SOURCES))} "
          f"refactored={refactored:.3f} unrelated={unrelated:.3f} threshold={THRESHOLD}")
    print(f"clusters={groups}")
    print("PASS")


if __name__ == "__main__":
    main()
