"""Task 468 - Implement streaming quantiles (Greenwald-Khanna outline). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Greenwald-Khanna ε-approximate quantile summary (SIGMOD 2001), batch merge-compress
variant: inserted values are buffered and every int(1/eps)+1 arrivals are merged
into the summary entries [value, g, delta] with band floor(2*eps*(n-1)); a fresh
value landing before an existing entry receives delta = entry.g + entry.delta - 1,
otherwise neighbours are absorbed while g_left + g_right + delta_right <= band.
QUANTILE(φ) walks the entries accumulating g and returns the last entry whose
g_sum + delta stays within int(φ*(n-1)+1) + int(eps*(n-1)), falling back to the
tracked minimum.

Checks against the exact sorted oracle: for every returned value v,
  rl = #{x < v}  <=  φ*N + ε*N    and    ru = #{x <= v}  >=  φ*N − ε*N,
for eps = 0.05, N = 2000, φ in {0, 0.05, ..., 1.0} over six stream patterns
(random, increasing, decreasing, constant, heavy duplicates, sawtooth); query(0)
returns the true minimum and the returned value is always a stream member. The
summary stays compressed: documented size bound (1/(2ε)+1)*(log2(2εn)+2) = 106
tuples, measured maximum 82 (decreasing stream).
"""
import math


class GreenwaldKhanna:
    def __init__(self, eps):
        self.eps = eps
        self.entries = []  # [value, g, delta]
        self.pending = []
        self.count = 0
        self.minimum = None

    def insert(self, value):
        self.pending.append(value)
        self.count += 1
        if self.minimum is None or value < self.minimum:
            self.minimum = value
        if self.count % (int(1.0 / self.eps) + 1) == 0:
            self.compress()

    def compress(self):
        threshold = int(2.0 * self.eps * (self.count - 1))
        fresh = [[v, 1, 0] for v in sorted(self.pending)]
        old = self.entries
        out = []
        i = j = 0
        nf, no = len(fresh), len(old)
        while i < nf or j < no:
            if i == nf:
                t = old[j]
                j += 1
                if j < no and t[1] + old[j][1] + old[j][2] <= threshold:
                    old[j][1] += t[1]
                    continue
                out.append(t)
            elif j == no:
                t = fresh[i]
                i += 1
                if i < nf and t[1] + fresh[i][1] + fresh[i][2] <= threshold:
                    fresh[i][1] += t[1]
                    continue
                out.append(t)
            elif fresh[i][0] < old[j][0]:
                t, u = fresh[i], old[j]
                if t[1] + u[1] + u[2] <= threshold:
                    u[1] += t[1]
                else:
                    t[2] = u[1] + u[2] - t[1]
                    out.append(t)
                i += 1
            else:
                t = old[j]
                j += 1
                if j < no and t[1] + old[j][1] + old[j][2] <= threshold:
                    old[j][1] += t[1]
                    continue
                out.append(t)
        self.entries = out
        self.pending = []

    def query(self, phi):
        if self.count == 0:
            return None
        if self.pending:
            self.compress()
        rank = int(phi * (self.count - 1) + 1)
        spread = int(self.eps * (self.count - 1))
        total = 0
        i = 0
        n = len(self.entries)
        while i < n:
            total += self.entries[i][1]
            if total + self.entries[i][2] > rank + spread:
                break
            i += 1
        if i == 0:
            return self.minimum
        return self.entries[i - 1][0]


def lcg_value(seed):
    return (seed * 75 + 74) % 65537


def streams():
    """Six deterministic stream patterns of length 2000."""
    seed = 20260116
    random_stream = []
    for _ in range(2000):
        seed = lcg_value(seed)
        random_stream.append(seed % 1000)
    dupes = []
    for _ in range(2000):
        seed = lcg_value(seed)
        dupes.append(seed % 5)
    return [
        ("random", random_stream),
        ("increasing", list(range(2000))),
        ("decreasing", list(range(1999, -1, -1))),
        ("constant", [42] * 2000),
        ("dupes", dupes),
        ("sawtooth", [i % 50 for i in range(2000)]),
    ]


def main():
    eps = 0.05
    phis = [i / 20.0 for i in range(21)]  # 0, 0.05, ..., 1.0
    max_size = 0
    for name, data in streams():
        sketch = GreenwaldKhanna(eps)
        for value in data:
            sketch.insert(value)
        n = len(data)
        exactly = sorted(data)
        for phi in phis:
            value = sketch.query(phi)
            assert value in data, (name, phi)
            lo = 0
            while lo < n and exactly[lo] < value:
                lo += 1
            hi = lo
            while hi < n and exactly[hi] <= value:
                hi += 1
            target = phi * n
            slack = eps * n
            assert lo <= target + slack + 1e-9, (name, phi, value, lo)  # ε-approx upper rank
            assert hi >= target - slack - 1e-9, (name, phi, value, hi)  # ε-approx lower rank
        assert sketch.query(0.0) == min(data), name
        max_size = max(max_size, len(sketch.entries))  # after queries force compression
    assert max_size <= (1.0 / (2 * eps) + 1) * (math.log2(2 * eps * 2000) + 2), max_size
    assert max_size == 82  # measured maximum compressed size (decreasing stream)

    # Empty sketch and tiny stream.
    empty = GreenwaldKhanna(0.01)
    assert empty.query(0.5) is None
    tiny = GreenwaldKhanna(0.01)
    for value in [5, 1, 9, 3, 7]:
        tiny.insert(value)
    assert tiny.query(0.0) == 1
    assert tiny.query(1.0) == 9
    assert tiny.query(0.5) in {3, 5}


if __name__ == "__main__":
    main()
    print("PASS")
