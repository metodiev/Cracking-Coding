"""Task 405 - Implement a minimal DFA for a fixed string set (conceptual). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Scope: the DFA recognizes exactly the finite set of words it was built from - no other string
is accepted. Only the *prefixes* of the set's words matter: the builder's states are exactly
the distinct non-empty prefixes plus the empty root, and any character that no prefix uses
leads to an explicit dead state. The alphabet is the sorted union of the words' characters;
characters outside it are documented to reject.

Minimization uses the classic table-filling (Moore) algorithm: mark every pair that differs
in acceptance, then repeatedly mark a pair when some character leads to an already marked
pair; inseparable pairs are merged with union-find. For {"he", "she", "his"} the accepting
states "he", "she", "his" merge into one class (nothing extends a complete word), so the
complete minimal DFA has 7 states: {""}, {"h"}, {"s"}, {"sh"}, {"hi"}, {"he","she","his"} and
one dead state. Note "h" does NOT merge with "sh": "h" still accepts the continuation "is"
("his") while "sh" only accepts "e". Dropping the dead state leaves the 6-state partial DFA
that tracks exactly the set's prefixes.

Checks: the enumerated language up to length 4 equals the key set exactly, a list of near-miss
strings is rejected, every state is reachable, and the minimized automaton has no pair of
distinguishable-by-nothing states (product-BFS equivalence test), i.e. it is minimal.
"""

def build_exact_match_dfa(words, alphabet):
    """Complete DFA whose language is exactly the given set (prefix-trie + dead state)."""
    states = [""]
    index = {"": 0}
    accept = [False]
    trans = [[-1] * len(alphabet)]
    for word in words:
        node = 0
        prefix = ""
        for ch in word:
            nxt = prefix + ch
            if nxt not in index:
                index[nxt] = len(states)
                states.append(nxt)
                accept.append(False)
                trans.append([-1] * len(alphabet))
            col = alphabet.index(ch)
            if trans[node][col] == -1:
                trans[node][col] = index[nxt]
            node = index[nxt]
            prefix = nxt
        accept[node] = True
    dead = len(states)
    states.append("(dead)")
    accept.append(False)
    trans.append([dead] * len(alphabet))
    for row in trans:
        for col in range(len(alphabet)):
            if row[col] == -1:
                row[col] = dead
    return states, index, trans, accept, dead


def minimize_dfa(trans, accept, alphabet_size):
    """Table-filling minimization; returns (class_of, new_trans, new_accept). Class 0 = start."""
    n = len(trans)
    marked = [[False] * n for _ in range(n)]
    for p in range(n):
        for q in range(p + 1, n):
            if accept[p] != accept[q]:
                marked[p][q] = True
    changed = True
    while changed:
        changed = False
        for p in range(n):
            for q in range(p + 1, n):
                if marked[p][q]:
                    continue
                for c in range(alphabet_size):
                    a, b = trans[p][c], trans[q][c]
                    if a == b:
                        continue
                    lo, hi = (a, b) if a < b else (b, a)
                    if marked[lo][hi]:
                        marked[p][q] = True
                        changed = True
                        break
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in range(n):
        for q in range(p + 1, n):
            if not marked[p][q]:
                rp, rq = find(p), find(q)
                if rp != rq:
                    parent[max(rp, rq)] = min(rp, rq)
    reps = sorted({find(i) for i in range(n)})
    remap = {rep: i for i, rep in enumerate(reps)}
    class_of = [remap[find(i)] for i in range(n)]
    new_trans = [[class_of[trans[reps[r]][c]] for c in range(alphabet_size)] for r in range(len(reps))]
    new_accept = [accept[reps[r]] for r in range(len(reps))]
    return class_of, new_trans, new_accept


def accepts(trans, accept, word, alphabet):
    alphabet_chars = set(alphabet)
    state = 0
    for ch in word:
        if ch not in alphabet_chars:      # documented: outside the alphabet, reject
            return False
        state = trans[state][alphabet.index(ch)]
    return accept[state]


def reachable_states(trans):
    seen = {0}
    stack = [0]
    while stack:
        state = stack.pop()
        for nxt in trans[state]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def all_pairs_distinguishable(trans, accept, alphabet_size):
    """Product-BFS equivalence test: minimal iff every reachable pair of states differs."""
    n = len(trans)
    for p in range(n):
        for q in range(p + 1, n):
            seen = {(p, q)}
            stack = [(p, q)]
            equivalent = True
            while stack and equivalent:
                a, b = stack.pop()
                if accept[a] != accept[b]:
                    equivalent = False
                    break
                for c in range(alphabet_size):
                    na, nb = trans[a][c], trans[b][c]
                    lo, hi = (na, nb) if na < nb else (nb, na)
                    if (lo, hi) not in seen:
                        seen.add((lo, hi))
                        stack.append((na, nb))
            if equivalent:
                return False
    return True


def dead_states(trans, accept):
    """Non-accepting classes with no escape: reject everything from here on."""
    return {state for state in range(len(trans))
            if not accept[state] and all(nxt == state for nxt in trans[state])}


def language_of_dfa(trans, accept, alphabet, max_len):
    accepted = set()
    frontier = [""]
    for _ in range(max_len + 1):
        nxt_frontier = []
        for word in frontier:
            if accepts(trans, accept, word, alphabet):
                accepted.add(word)
            if len(word) < max_len:
                nxt_frontier.extend(word + ch for ch in alphabet)
        frontier = nxt_frontier
    return accepted


def main():
    alphabet = "ehis"
    words = ["he", "she", "his"]
    states, index, trans, accept, dead = build_exact_match_dfa(words, alphabet)

    # states = distinct prefixes (only the set's prefixes matter) + dead
    prefixes = {""}
    for word in words:
        for k in range(1, len(word) + 1):
            prefixes.add(word[:k])
    assert set(states[:-1]) == prefixes
    assert len(states) == len(prefixes) + 1 and states[dead] == "(dead)"
    assert len(reachable_states(trans)) == len(states)

    for word in words:
        assert accepts(trans, accept, word, alphabet), word
    rejections = ["", "h", "s", "e", "i", "heh", "hehe", "sh", "shi", "shee", "shes",
                  "hiss", "hishe", "hhi", "hh", "sheh", "hise", "hh"]
    for word in rejections:
        assert not accepts(trans, accept, word, alphabet), word
    for word in ["hey", "his!", "she?"]:
        assert not accepts(trans, accept, word, alphabet), word

    assert language_of_dfa(trans, accept, alphabet, 4) == set(words)

    class_of, minimal_trans, minimal_accept = minimize_dfa(trans, accept, len(alphabet))
    assert len(minimal_trans) == 7, len(minimal_trans)
    assert sum(minimal_accept) == 1
    assert reachable_states(minimal_trans) == set(range(len(minimal_trans)))
    assert all_pairs_distinguishable(minimal_trans, minimal_accept, len(alphabet))

    assert language_of_dfa(minimal_trans, minimal_accept, alphabet, 4) == set(words)
    for word in words:
        assert accepts(minimal_trans, minimal_accept, word, alphabet), word
    for word in rejections:
        assert not accepts(minimal_trans, minimal_accept, word, alphabet), word

    # documented merges: the accepting ends merge; "h" stays distinct from "sh" and "hi"
    assert class_of[index["he"]] == class_of[index["she"]] == class_of[index["his"]]
    assert class_of[index["h"]] != class_of[index["sh"]]     # "h"+"is" = "his", "sh"+"is" fails
    assert class_of[index["sh"]] != class_of[index["hi"]]
    assert class_of[index["h"]] != class_of[index["s"]]
    assert class_of[index[""]] != class_of[dead]
    assert len(set(class_of)) == 7

    # dropping the single dead state leaves the 6-state partial DFA of the prefix set
    assert dead_states(minimal_trans, minimal_accept) == {class_of[dead]}
    assert len(minimal_trans) - 1 == 6

    # second fixed set: {"ab", "b", "ba"} over {"a","b"} -> 5 states
    alphabet2 = "ab"
    words2 = ["ab", "b", "ba"]
    states2, index2, trans2, accept2, dead2 = build_exact_match_dfa(words2, alphabet2)
    assert accepts(trans2, accept2, "b", alphabet2)
    assert accepts(trans2, accept2, "ab", alphabet2)
    assert accepts(trans2, accept2, "ba", alphabet2)
    for word in ["", "a", "bb", "aba", "abb", "bab", "baa", "b b"]:
        assert not accepts(trans2, accept2, word, alphabet2), word
    assert language_of_dfa(trans2, accept2, alphabet2, 4) == set(words2)
    class_of2, minimal2_trans, minimal2_accept = minimize_dfa(trans2, accept2, len(alphabet2))
    assert len(minimal2_trans) == 5, len(minimal2_trans)
    assert all_pairs_distinguishable(minimal2_trans, minimal2_accept, len(alphabet2))
    assert language_of_dfa(minimal2_trans, minimal2_accept, alphabet2, 4) == set(words2)
    assert class_of2[index2["ab"]] == class_of2[index2["ba"]]
    assert class_of2[index2["ab"]] != class_of2[index2["b"]]
    assert class_of2[index2["a"]] != class_of2[dead2]


if __name__ == "__main__":
    main()
    print("PASS")
