"""Task 257 - Top K frequent words (heap + lexicographic tie). (Tier: Intermediate, CTCI Ch 10)

LeetCode 692 ordering rule: more frequent words first; on equal counts the lexicographically
smaller word comes first. The bounded heap holds the k best words, ordered worst-first so
the root is the first eviction candidate: "worse" means lower count, or - on a count tie -
the lexicographically larger word. Time O(n + d log k) for d distinct words.
"""

import heapq


class _WorstEntry:
    """Heap entry ordered so that the *smallest* entry is the worst candidate."""

    __slots__ = ("count", "word")

    def __init__(self, count, word):
        self.count = count
        self.word = word

    def __lt__(self, other):
        if self.count != other.count:
            return self.count < other.count
        return self.word > other.word


def top_k_frequent_words(words, k):
    if k <= 0:
        return []
    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1
    heap = []
    for word, count in freq.items():
        entry = _WorstEntry(count, word)
        if len(heap) < k:
            heapq.heappush(heap, entry)
        elif heap[0] < entry:
            heapq.heapreplace(heap, entry)
    return [e.word for e in sorted(heap, key=lambda e: (-e.count, e.word))]


def main():
    classic = ["i", "love", "leetcode", "i", "love", "coding"]  # i:2 love:2 leetcode:1 coding:1
    assert top_k_frequent_words(classic, 1) == ["i"]
    assert top_k_frequent_words(classic, 2) == ["i", "love"]
    assert top_k_frequent_words(classic, 3) == ["i", "love", "coding"]
    assert top_k_frequent_words(classic, 4) == ["i", "love", "coding", "leetcode"]
    assert top_k_frequent_words(classic, 9) == ["i", "love", "coding", "leetcode"]
    assert top_k_frequent_words(classic, 0) == []

    # is:3 the:3 sunny:2 day:1
    other = ["the", "day", "is", "sunny", "the", "the", "sunny", "is", "is"]
    assert top_k_frequent_words(other, 1) == ["is"]
    assert top_k_frequent_words(other, 2) == ["is", "the"]
    assert top_k_frequent_words(other, 4) == ["is", "the", "sunny", "day"]

    assert top_k_frequent_words(["a", "aa", "aaa"], 2) == ["a", "aa"]
    assert top_k_frequent_words(["a", "aa", "aaa"], 3) == ["a", "aa", "aaa"]
    # a:2 b:2 c:2 d:1 - eviction ties resolved lexicographically
    assert top_k_frequent_words(["b", "b", "a", "a", "c", "c", "d"], 1) == ["a"]
    assert top_k_frequent_words(["b", "b", "a", "a", "c", "c", "d"], 2) == ["a", "b"]
    assert top_k_frequent_words(["b", "b", "a", "a", "c", "c", "d"], 3) == ["a", "b", "c"]
    assert top_k_frequent_words(["only"], 1) == ["only"]
    assert top_k_frequent_words([], 3) == []


if __name__ == "__main__":
    main()
    print("PASS")
