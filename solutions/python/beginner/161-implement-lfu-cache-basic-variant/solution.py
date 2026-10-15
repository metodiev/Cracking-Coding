
"""Task 161 - Implement LFU cache (basic variant). (Tier: Beginner, CTCI Ch 7)

Evict the least frequently used key; ties break by least recently used.
Basic variant: eviction scans for min (freq, seq) — O(n) eviction, O(1) get/put amortized.
"""


class LFUCache:
    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._data = {}          # key -> [value, freq, seq]
        self._tick = 0

    def get(self, key):
        if key not in self._data:
            return -1
        entry = self._data[key]
        entry[1] += 1            # bump frequency
        entry[2] = self._tick    # refresh recency
        self._tick += 1
        return entry[0]

    def put(self, key, value):
        if key in self._data:
            entry = self._data[key]
            entry[0] = value
            entry[1] += 1
            entry[2] = self._tick
            self._tick += 1
            return
        if len(self._data) >= self.capacity:
            victim = min(self._data.items(), key=lambda kv: (kv[1][1], kv[1][2]))[0]
            del self._data[victim]
        self._data[key] = [value, 1, self._tick]
        self._tick += 1


def main():
    cache = LFUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(1) == 1            # freq(1) = 2
    cache.put(3, 3)                     # evicts key 2 (freq 1)
    assert cache.get(2) == -1
    assert cache.get(3) == 3            # freq(3) = 2
    cache.put(4, 4)                     # tie freq 2 vs 2 -> evict LRU key 1
    assert cache.get(1) == -1
    assert cache.get(3) == 3
    assert cache.get(4) == 4

    # updating an existing key refreshes it and keeps frequency climbing
    c2 = LFUCache(1)
    c2.put(9, 1)
    c2.put(9, 2)
    assert c2.get(9) == 2
    c2.put(8, 8)                        # evicts 9 (the only key)
    assert c2.get(9) == -1 and c2.get(8) == 8


if __name__ == "__main__":
    main()
    print("PASS")
