"""Task 268 - Word ladder shortest transformation (BFS). (Tier: Intermediate, CTCI Ch 4)

Shortest transformation sequence LENGTH (number of words, begin and end included)
between begin and end, where each step changes exactly one letter and every intermediate
word must be in the word list. BFS over the implicit one-letter-difference graph with a
wildcard bucket index (h*t -> [hot, hit, ...]) built once, so all neighbours of a word
are found in O(|word|) instead of scanning the whole list; O(N * L) build + O(N * L)
search for N words of length L.

Return value: 0 when no sequence exists (documented choice; LeetCode 127 agrees).
Special cases (documented): end not in the word list -> 0; begin == end -> 1 (already
there); mismatched word lengths -> 0; duplicate word-list entries collapse to one word.

ladder_length_oracle rescans the whole list for every node and is used only for
cross-checks.
"""

from collections import deque, defaultdict


def ladder_length(begin, end, word_list):
    words = set(word_list)
    if end not in words or len(begin) != len(end):
        return 0
    if begin == end:
        return 1
    length = len(begin)
    buckets = defaultdict(list)
    for word in words:
        if len(word) == length:
            for i in range(length):
                buckets[word[:i] + "*" + word[i + 1:]].append(word)
    queue = deque([begin])
    seen = {begin}
    steps = 1
    while queue:
        for _ in range(len(queue)):
            word = queue.popleft()
            for i in range(length):
                pattern = word[:i] + "*" + word[i + 1:]
                for nxt in buckets.get(pattern, ()):
                    if nxt == end:
                        return steps + 1
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
        steps += 1
    return 0


def ladder_length_oracle(begin, end, word_list):
    """Naive BFS that scans every word for a one-letter difference; cross-check only."""
    words = set(word_list)
    if end not in words or len(begin) != len(end):
        return 0
    if begin == end:
        return 1
    queue = deque([(begin, 1)])
    seen = {begin}
    while queue:
        word, dist = queue.popleft()
        for candidate in words:
            if candidate in seen or len(candidate) != len(word):
                continue
            if sum(a != b for a, b in zip(word, candidate)) == 1:
                if candidate == end:
                    return dist + 1
                seen.add(candidate)
                queue.append((candidate, dist + 1))
    return 0


def main():
    assert ladder_length("hit", "cog", ["hot", "dot", "dog", "lot", "log", "cog"]) == 5   # hit-hot-dot-dog-cog
    assert ladder_length("hit", "cog", ["hot", "dot", "dog", "lot", "log"]) == 0           # end not in list
    assert ladder_length("a", "c", ["a", "b", "c"]) == 2
    assert ladder_length("hit", "hot", ["hot", "dot", "dog"]) == 2
    assert ladder_length("hot", "hot", ["hot"]) == 1                                       # same word
    assert ladder_length("ab", "abc", ["abc"]) == 0                                        # mismatched lengths
    assert ladder_length("abc", "ab", ["ab"]) == 0
    assert ladder_length("lost", "cost", ["most", "fost", "cost", "cast"]) == 2            # lost -> cost
    assert ladder_length("ab", "cd", ["ac", "bd"]) == 0                                    # unreachable
    assert ladder_length("hit", "cog", ["hot", "hot", "dot", "dog", "lot", "log", "cog"]) == 5   # duplicates collapse
    assert ladder_length("hot", "hot", ["dog"]) == 0                                       # end not in list wins
    assert ladder_length("aaaa", "aaab", ["aaaa", "aaab"]) == 2

    import random

    rng = random.Random(268)
    for _ in range(300):                            # cross-check vs the naive-scan BFS
        pool = sorted({"".join(rng.choice("abc") for _ in range(3)) for _ in range(rng.randint(1, 6))})
        word_list = list(pool)
        if rng.random() < 0.5:
            word_list += pool[:1]                   # duplicate entry
        begin = "".join(rng.choice("abc") for _ in range(3))
        end = rng.choice(pool) if pool and rng.random() < 0.5 else "".join(rng.choice("abc") for _ in range(3))
        assert ladder_length(begin, end, word_list) == ladder_length_oracle(begin, end, word_list)


if __name__ == "__main__":
    main()
    print("PASS")
