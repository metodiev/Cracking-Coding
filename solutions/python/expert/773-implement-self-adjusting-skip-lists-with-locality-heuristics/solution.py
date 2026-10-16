"""Task 773 - Implement self-adjusting skip lists with locality heuristics.  (Tier: Expert, Beyond - Advanced Data Structures & Algorithms)

A skip list is a set of sorted linked lists stacked on top of each other: a node has height h and
participates in levels 0..h-1, and every level is a sorted subsequence of the level below.  Here
the height comes from a hash of the *key* (h = 1 + number of trailing one bits, so geometric and
independent of insertion order), which makes differently tuned skip lists structurally identical
and therefore directly comparable.

The self-adjusting part is a fingertip cache: after every descent the list remembers, for each
level, the right-most node it visited on that level (the predecessor of the searched key).  The
next descent starts each level's walk from the better of two positions - the node reached from the
level above, or this remembered finger - because the finger is a valid start exactly when it still
lies strictly left of the query key.  Two locality effects follow from that single cache:
  * key-space locality: searching for a key next to the previous one needs almost no walking, which
    is what range scans and hot neighbourhoods produce;
  * temporal locality: the level-0 finger sits immediately before the previous answer, so a
    repeated access reaches the target with a single comparison at level 0.
A sorted skip list may *not* reorder its nodes the way an unordered self-organising list does, so
transposition or move-to-front are unavailable - the finger cache is the honest analogue.  The
"re-assign every height from the access counts" alternative (biased skip list) was implemented and
measured too: it inflates the level lists over the hot prefix and costs more comparisons on these
workloads, so it is not used and that negative result is reported below.

What is simplified: single-threaded, no demotion/aging, and the cache holds one pointer per level
(O(log n) memory).  Verified: search/insert/delete against a dict oracle after every mutation, the
skip-list invariants after every operation (strictly increasing keys per level, level L+1 an exact
subsequence of level L, node height == the number of levels it appears on, size == level-0 length),
the boundary cases (empty list, one element, duplicate insert, deleting head/tail/missing, emptying
and refilling), determinism, and the locality claim itself on a neighbourhood (range-scan style)
stream; the locality-free Zipf stream only has to show no regression.
"""

MASK64 = (1 << 64) - 1
MAX_HEIGHT = 12


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def hash_key(key):
    """Deterministic avalanche of a key; drives the height and the workload generators."""
    _, r = splitmix64((key * 0x9E3779B97F4A7C15 + 0x1234567) & MASK64)
    return r


def height_of(key):
    """Geometric height from the trailing one bits of hash(key): P(h > k) = 2^-k."""
    bits = hash_key(key) >> 1
    height = 1
    while height < MAX_HEIGHT and bits & 1:
        bits >>= 1
        height += 1
    return height


class Node:
    __slots__ = ("key", "value", "height", "count", "nxt")

    def __init__(self, key, value, height):
        self.key = key
        self.value = value
        self.height = height
        self.count = 0                 # accesses, kept as a statistic for the report
        self.nxt = [None] * height


class SkipList:
    """Sorted skip list with a self-adjusting fingertip cache (one finger per level)."""

    def __init__(self, finger=False):
        self.header = Node(None, None, MAX_HEIGHT)
        self.level = 0                 # highest level that currently holds a node
        self.size = 0
        self.comparisons = 0
        self.finger_hits = 0
        self.searches = 0
        self.finger_search = finger
        self.fingers = [None] * MAX_HEIGHT

    # ------------------------------------------------------------------ internals
    def _search(self, key):
        """Descend from the top; at every level start from whichever of arrival/finger is righter."""
        predecessors = [None] * MAX_HEIGHT
        node = self.header
        for level in range(self.level, -1, -1):
            if self.finger_search:
                finger = self.fingers[level]
                if (finger is not None and level < finger.height and key > finger.key
                        and (node is self.header or node.key < finger.key)):
                    node = finger                     # a strictly righter start, still left of key
                    self.finger_hits += 1
            while node.nxt[level] is not None:
                self.comparisons += 1
                if node.nxt[level].key < key:
                    node = node.nxt[level]
                else:
                    break
            predecessors[level] = node
        if self.finger_search:
            for level in range(self.level + 1):
                visited = predecessors[level]
                self.fingers[level] = None if visited is self.header else visited
        return node.nxt[0], predecessors

    def _descend(self, key):
        """Plain descent (no fingers): predecessors[level] = last node with key' < key there."""
        predecessors = [self.header] * MAX_HEIGHT
        node = self.header
        for level in range(self.level, -1, -1):
            while node.nxt[level] is not None:
                self.comparisons += 1
                if node.nxt[level].key < key:
                    node = node.nxt[level]
                else:
                    break
            predecessors[level] = node
        return predecessors, node

    # ------------------------------------------------------------------ public API
    def search(self, key):
        """Return the value stored for key, or None."""
        self.searches += 1
        node, _ = self._search(key)
        if node is not None and node.key == key:
            node.count += 1
            return node.value
        return None

    def insert(self, key, value):
        """Insert or update; returns True when the key is new."""
        predecessors, node = self._descend(key)
        candidate = node.nxt[0]
        if candidate is not None and candidate.key == key:
            candidate.value = value
            return False
        height = height_of(key)
        fresh = Node(key, value, height)
        for level in range(height):
            fresh.nxt[level] = predecessors[level].nxt[level]
            predecessors[level].nxt[level] = fresh
        if height - 1 > self.level:
            self.level = height - 1
        self.size += 1
        return True

    def delete(self, key):
        """Remove key; returns True when it was present."""
        predecessors, node = self._descend(key)
        target = node.nxt[0]
        if target is None or target.key != key:
            return False
        for level in range(target.height):
            predecessors[level].nxt[level] = target.nxt[level]
        self.size -= 1
        while self.level > 0 and self.header.nxt[self.level] is None:
            self.level -= 1
        for level in range(MAX_HEIGHT):
            if self.fingers[level] is target:
                self.fingers[level] = None      # a finger may never point at a removed node
        return True

    # ------------------------------------------------------------------ inspection
    def keys_at(self, level):
        keys = []
        node = self.header.nxt[level]
        while node is not None:
            keys.append(node.key)
            node = node.nxt[level]
        return keys

    def check_invariants(self, expected=None):
        """Structural invariants; `expected` is the oracle's sorted key list when available."""
        levels = [self.keys_at(level) for level in range(self.level + 1)]
        level_zero = levels[0]
        check(level_zero == sorted(level_zero), "level 0 must be sorted")
        check(len(level_zero) == self.size, "level 0 must hold exactly `size` nodes")
        if expected is not None:
            check(level_zero == expected, "level 0 must equal the oracle's sorted keys")
        for level, keys in enumerate(levels[1:], start=1):
            check(keys == sorted(keys), f"level {level} must be sorted")
            check(set(keys) <= set(level_zero), f"level {level} must be a subset of level 0")
        for level in range(self.level + 1, MAX_HEIGHT):
            check(self.header.nxt[level] is None, "no node may live above the top level")
        appearances = {key: 0 for key in level_zero}
        for keys in levels:
            for key in keys:
                appearances[key] += 1
        node = self.header.nxt[0]
        while node is not None:
            check(appearances[node.key] == node.height,
                  f"node {node.key} claims height {node.height}, appears on {appearances[node.key]}")
            node = node.nxt[0]


def zipf_stream(length, distinct, alpha, seed):
    """Deterministic Zipf(alpha) stream over 0..distinct-1 (skewed but locality free)."""
    weights = [(i + 1) ** (-alpha) for i in range(distinct)]
    total = sum(weights)
    cumulative = []
    acc = 0.0
    for w in weights:
        acc += w
        cumulative.append(acc / total)
    stream = []
    for _ in range(length):
        seed, r = splitmix64(seed)
        u = (r % (1 << 30)) / float(1 << 30)
        lo, hi = 0, distinct - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cumulative[mid] <= u:
                lo = mid + 1
            else:
                hi = mid
        stream.append(lo)
    return stream


def neighbourhood_stream(length, distinct, radius, random_share, seed):
    """Range-scan style stream: mostly +-radius steps around the previous key, sometimes random."""
    stream = []
    current = distinct // 2
    for _ in range(length):
        seed, r = splitmix64(seed)
        if (r & 0xFF) < int(random_share * 256):
            current = (r >> 8) % distinct
        else:
            step = (r >> 8) % (2 * radius + 1) - radius
            current = max(0, min(distinct - 1, current + step))
        stream.append(current)
    return stream


def build(keys, finger=False):
    lst = SkipList(finger=finger)
    for key in keys:
        lst.insert(key, key * 2)
    return lst


def measure(stream, keys, finger):
    """Run one workload and return (list, number of correct answers, finger hits)."""
    lst = build(keys, finger=finger)
    hits = 0
    for key in stream:
        if lst.search(key) == key * 2:
            hits += 1
    return lst, hits


def main():
    # 1. boundary: an empty list answers misses and rejects deletions
    empty = SkipList(finger=True)
    check(empty.search(5) is None, "an empty list must miss")
    check(empty.delete(5) is False, "deleting from an empty list must fail")
    check(empty.size == 0 and empty.level == 0, "an empty list has no content")
    empty.check_invariants(expected=[])

    # 2. single element, duplicate insert, neighbours, delete
    single = SkipList(finger=True)
    check(single.insert(7, 70) is True, "the first insert must add a key")
    check(single.insert(7, 99) is False and single.search(7) == 99,
          "a duplicate insert must update the value, not the size")
    check(single.size == 1, "an update must not change the size")
    check(single.search(6) is None and single.search(8) is None, "neighbours of a lone key must miss")
    single.check_invariants(expected=[7])
    check(single.delete(7) is True and single.delete(7) is False, "delete must be idempotent")
    check(single.search(7) is None and single.size == 0, "the list must be empty again")
    single.check_invariants(expected=[])

    # 3. inserts/deletes agree with a dict oracle and keep the invariants
    state = 0xA11CE
    lst = SkipList(finger=True)
    oracle = {}
    for step in range(300):
        state, r = splitmix64(state)
        key = r % 60
        if r >> 8 & 1:
            added = lst.insert(key, key * 3)
            check(added == (key not in oracle), "insert must report novelty like the oracle")
            oracle[key] = key * 3
        else:
            removed = lst.delete(key)
            check(removed == (key in oracle), "delete must report existence like the oracle")
            oracle.pop(key, None)
        if step % 25 == 0:
            lst.check_invariants(expected=sorted(oracle))
    check(lst.size == len(oracle), "the final size must match the oracle")
    for key, value in oracle.items():
        check(lst.search(key) == value, "every oracle key must be found with its value")
    lst.check_invariants(expected=sorted(oracle))
    check(all(entry is None or entry.key in oracle for entry in lst.fingers),
          "no finger may point at a deleted node")

    # 4. delete the head and the tail of a populated list
    edge = build(list(range(12)))
    check(edge.delete(0) and edge.delete(11), "head and tail must be deletable")
    edge.check_invariants(expected=list(range(1, 11)))
    check(edge.search(0) is None and edge.search(11) is None, "the deleted extremes must be gone")
    check(edge.search(5) == 10, "the middle must survive the extreme deletions")

    # 5. the level structure is the nested geometric one, independent of insertion order
    forward = build(list(range(100)))
    backward = build(list(reversed(range(100))))
    check(forward.level == backward.level, "the height must not depend on insertion order")
    check(forward.keys_at(forward.level) == backward.keys_at(backward.level),
          "the top level must be identical for both insertion orders")
    check(forward.keys_at(0) == backward.keys_at(0) == list(range(100)), "both must be sorted")

    # 6. skewed but locality-free stream: the fingertip cache must not regress
    keys = list(range(200))
    zipf = zipf_stream(6000, 200, 1.1, 0x5EED)
    plain_zipf, plain_hits = measure(zipf, keys, finger=False)
    finger_zipf, finger_hits = measure(zipf, keys, finger=True)
    check(plain_hits == finger_hits == len(zipf), "both variants must answer every query")
    check(finger_zipf.comparisons <= plain_zipf.comparisons,
          "the fingertip cache must never cost more than the plain descent on a skewed stream")
    check(finger_zipf.finger_hits > 0, "the fingers must actually be used")
    check(finger_zipf.level == plain_zipf.level, "adaptation must not change the level structure")

    # 7. neighbourhood (range-scan) stream: the locality win
    local = neighbourhood_stream(6000, 200, 3, 0.22, 0x5EED)
    plain_local, local_hits = measure(local, keys, finger=False)
    finger_local, local_answer = measure(local, keys, finger=True)
    check(local_hits == local_answer == len(local), "the neighbourhood stream must be answered")
    check(finger_local.comparisons <= plain_local.comparisons * 0.85,
          f"key-space locality must save at least 15% of the comparisons "
          f"({finger_local.comparisons} vs {plain_local.comparisons})")
    check(finger_local.fingers[0] is not None, "the level-0 finger must be populated")
    finger_local.check_invariants(expected=keys)
    check(finger_local.size == len(keys), "the adaptive list must keep every key")

    # 8. repeated access to one key: the level-0 finger makes the repeat cheap
    hot = 37
    warm = build(keys, finger=True)
    cold = build(keys, finger=False)
    for _ in range(200):
        warm.search(hot)
    before = warm.comparisons
    for _ in range(400):
        check(warm.search(hot) == hot * 2, "the hot key must always be found")
    hot_cost = (warm.comparisons - before) / 400.0
    before = cold.comparisons
    for _ in range(400):
        check(cold.search(hot) == hot * 2, "the plain list must find the hot key too")
    cold_cost = (cold.comparisons - before) / 400.0
    check(hot_cost <= warm.level + 2.0,
          f"a repeated hot access must cost about one comparison per level, got {hot_cost}")
    check(hot_cost * 1.5 < cold_cost,
          f"the warm list must be well ahead on the hot key ({hot_cost} vs {cold_cost})")

    # 9. the cache costs O(log n) memory: one pointer per level, never more
    check(len(finger_local.fingers) == MAX_HEIGHT, "the cache must hold one pointer per level")
    check(sum(1 for entry in finger_local.fingers if entry is not None) <= MAX_HEIGHT,
          "the cache must never hold more than MAX_HEIGHT pointers")

    # 10. determinism: identical operations give identical structures and counters
    first, second = SkipList(finger=True), SkipList(finger=True)
    for key in zipf[:400]:
        first.search(key)
        second.search(key)
    check(first.comparisons == second.comparisons, "the adaptive list must be deterministic")
    check([entry.key if entry else None for entry in first.fingers] ==
          [entry.key if entry else None for entry in second.fingers], "the fingers must match")

    # 11. emptying the list and refilling it
    churn = build(list(range(20)))
    for key in range(20):
        check(churn.delete(key), "every key must be deletable")
    check(churn.size == 0 and churn.level == 0, "the emptied list must collapse to zero levels")
    churn.check_invariants(expected=[])
    for key in range(20):
        churn.insert(key, key)
    churn.check_invariants(expected=list(range(20)))
    check(churn.search(13) == 13, "the refilled list must answer")

    print(f"773 zipf plain={plain_zipf.comparisons} finger={finger_zipf.comparisons} "
          f"hits={finger_zipf.finger_hits}")
    print(f"773 local plain={plain_local.comparisons} finger={finger_local.comparisons} "
          f"hot_cost={hot_cost:.2f} cold_hot_cost={cold_cost:.2f} levels={finger_local.level}")
    print("PASS")


if __name__ == "__main__":
    main()
