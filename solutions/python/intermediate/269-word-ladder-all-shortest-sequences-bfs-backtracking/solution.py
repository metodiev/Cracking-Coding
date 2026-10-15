"""Task 269 - Word ladder all shortest sequences (BFS + backtracking). (Tier: Intermediate, CTCI Ch 4)

All shortest transformation sequences from begin to end (each step changes one letter,
every intermediate word must be in the word list). BFS over the wildcard bucket index
assigns each word its minimum distance; the DFS backtracking step then walks the LAYERED
DAG, following only edges u -> v with dist[v] == dist[u] + 1. That layering rule keeps
every shortest path exactly once (dist strictly increases along a path, so the DFS never
revisits a word within one path) and prunes all non-shortest continuations.

Determinism (documented): the returned list of sequences is sorted element-wise in
lexicographic order, so hash-iteration order never leaks into the result. Boundaries:
[] when end is not in the word list, when the lengths differ, or when end is
unreachable; [[begin]] when begin == end (and end is in the list).

find_ladders_brute enumerates all simple paths and keeps the shortest ones; it is used
only for cross-checks.
"""

from collections import deque, defaultdict


def find_ladders(begin, end, word_list):
    words = set(word_list)
    if end not in words or len(begin) != len(end):
        return []
    if begin == end:
        return [[begin]]
    length = len(begin)
    buckets = defaultdict(list)
    for word in words:
        if len(word) == length:
            for i in range(length):
                buckets[word[:i] + "*" + word[i + 1:]].append(word)

    dist = {begin: 0}                               # BFS: minimum distance per word
    queue = deque([begin])
    while queue:
        word = queue.popleft()
        for i in range(length):
            for nxt in buckets.get(word[:i] + "*" + word[i + 1:], ()):
                if nxt not in dist:
                    dist[nxt] = dist[word] + 1
                    queue.append(nxt)
    if end not in dist:
        return []

    target = dist[end]
    result = []
    path = [begin]

    def dfs(word):                                  # backtrack along the layered DAG
        if word == end:
            result.append(list(path))
            return
        if dist[word] >= target:
            return
        for i in range(length):
            for nxt in buckets.get(word[:i] + "*" + word[i + 1:], ()):
                if dist.get(nxt) == dist[word] + 1:
                    path.append(nxt)
                    dfs(nxt)
                    path.pop()

    dfs(begin)
    result.sort()
    return result


def find_ladders_brute(begin, end, word_list):
    """Enumerate all simple paths, keep the shortest; cross-check only."""
    words = set(word_list)
    if end not in words or len(begin) != len(end):
        return []
    if begin == end:
        return [[begin]]

    def neighbours(word):
        return [v for v in words if len(v) == len(word) and sum(a != b for a, b in zip(word, v)) == 1]

    found = []

    def dfs(word, path, seen):
        if word == end:
            found.append(list(path))
            return
        if len(path) > len(words) + 1:              # simple paths cannot exceed this
            return
        for nxt in neighbours(word):
            if nxt not in seen:
                seen.add(nxt)
                path.append(nxt)
                dfs(nxt, path, seen)
                path.pop()
                seen.remove(nxt)

    dfs(begin, [begin], {begin})
    if not found:
        return []
    best = min(len(p) for p in found)
    return sorted(p for p in found if len(p) == best)


def main():
    assert find_ladders("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]) == [
        ["hit", "hot", "dot", "dog", "cog"],
        ["hit", "hot", "lot", "log", "cog"],
    ]
    assert find_ladders("red", "tax", ["ted", "tex", "red", "tax", "tad", "den", "rex", "pee"]) == [
        ["red", "rex", "tex", "tax"],
        ["red", "ted", "tad", "tax"],
        ["red", "ted", "tex", "tax"],
    ]
    assert find_ladders("ab", "ba", ["bb", "aa", "ba"]) == [["ab", "aa", "ba"], ["ab", "bb", "ba"]]
    assert find_ladders("hit", "cog", ["hot", "dot", "dog", "lot", "log"]) == []      # unreachable
    assert find_ladders("hot", "hot", ["hot"]) == [["hot"]]                           # same word
    assert find_ladders("a", "c", ["a", "b", "c"]) == [["a", "c"]]                   # direct one-letter hop
    assert find_ladders("ab", "abc", ["abc"]) == []                                   # mismatched lengths
    assert find_ladders("hot", "hot", ["dog"]) == []                                  # end not in list wins
    assert find_ladders("lost", "cost", ["most", "fost", "cost", "cast"]) == [["lost", "cost"]]

    import random

    rng = random.Random(269)
    for _ in range(200):                            # cross-check vs the path-enumeration brute
        pool = sorted({"".join(rng.choice("abc") for _ in range(3)) for _ in range(rng.randint(1, 6))})
        word_list = list(pool)
        begin = "".join(rng.choice("abc") for _ in range(3))
        end = rng.choice(pool) if pool and rng.random() < 0.5 else "".join(rng.choice("abc") for _ in range(3))
        assert find_ladders(begin, end, word_list) == find_ladders_brute(begin, end, word_list)


if __name__ == "__main__":
    main()
    print("PASS")
