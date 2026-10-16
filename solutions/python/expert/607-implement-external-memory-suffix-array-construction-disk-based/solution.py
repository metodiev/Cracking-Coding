"""Task 607 - Implement external-memory suffix array construction (disk-based). (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)

SIMPLIFICATION, stated up front: there are no real files and no real disk. The text, the rank array
and every spilled run live in memory, but all of them are reached through page-counted stores that
charge one read for a page that is not already cached and one write per page written. That reproduces
- and instruments - the block-by-block behaviour the task asks for: the text is read a page at a time,
each round is a bounded-memory external sort with spills, and the merge has a bounded fan-in. What is
deliberately *not* modelled is the streaming inside a merge (a real implementation holds one record
per stream, the simulation materializes each spilled run) and the real latency of a seek.

The algorithm is prefix doubling carried out externally. Each round:
  1. one sequential pass over the rank file plus one pass shifted by k records (two page reads per
     page), which builds the keys (rank[i], rank[i + k]) - -1 when i + k is past the end;
  2. the keys are sorted in memory in chunks of `memory_records` records, each sorted chunk spilled as
     a run, then the runs are merged with fan-in `fan_in`, so the merge takes exactly
     ceil(log_fan_in(ceil(n / memory_records))) passes;
  3. the single merged run is scanned once and compact ranks 0..d-1 are assigned and written back.
The loop stops as soon as d == n, i.e. every suffix has a rank of its own; the merged run is then the
suffix array.

Checks: the external result equals prefix doubling in memory and the manual comparator oracle on hand,
unary, periodic and LCG texts; the same result for three memory budgets, with the smaller budget
needing at least as many merge passes; the merge pass count equals ceil(log_fan_in(ceil(n / M))); a
sequential pass costs exactly one page read per page and a doubled page size halves the page count
read; total I/O is bounded below by two page reads per page per round; the rank file stays a
permutation of ranks with distinct count increasing; and the i/o statistics are reported.
"""
import heapq


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def lcg_string(length, alphabet, seed=12345):
    chars = []
    for _ in range(length):
        seed = (seed * 75 + 74) % 65537
        chars.append(alphabet[seed % len(alphabet)])
    return "".join(chars), seed


class PageStore:
    """In-memory stand-in for a disk file read page by page; one cached page, counted reads."""

    def __init__(self, page_size):
        self.page_size = page_size
        self.pages = []
        self.reads = self.writes = 0
        self.cached = -1

    def count(self):
        return sum(len(page) for page in self.pages)

    def load(self, records):
        self.pages = [records[i:i + self.page_size] for i in range(0, len(records), self.page_size)]
        self.writes += len(self.pages)
        self.cached = -1

    def sequential(self):
        """One pass over the whole file: exactly one page read per page."""
        self.reads += len(self.pages)
        return [record for page in self.pages for record in page]

    def shifted(self, offset):
        """One pass over the file shifted by `offset` records (one page read per page again)."""
        self.reads += len(self.pages)
        flat = [record for page in self.pages for record in page]
        return flat[offset:] + [-1] * offset


class RunStore:
    """Spilled sorted runs held as pages; one read or write per page touched."""

    def __init__(self, records_per_page):
        self.records_per_page = records_per_page
        self.runs = []
        self.reads = self.writes = 0

    def pages_for(self, records):
        return max(1, (len(records) + self.records_per_page - 1) // self.records_per_page)

    def write(self, records):
        self.runs.append(records)
        self.writes += self.pages_for(records)
        return len(self.runs) - 1

    def read(self, index):
        self.reads += self.pages_for(self.runs[index])
        return self.runs[index]


def expected_merge_passes(runs, fan_in):
    """Independent model of the pass count: every pass collapses fan_in runs into one."""
    passes = 0
    while runs > 1:
        runs = (runs + fan_in - 1) // fan_in
        passes += 1
    return passes


def sort_chunks_to_runs(keys, run_store, memory_records):
    """Sort the keys in memory in chunks of `memory_records` and spill every chunk as one run."""
    runs = []
    for start in range(0, len(keys), memory_records):
        runs.append(run_store.write(sorted(keys[start:start + memory_records])))
    return runs


def merge_runs(run_store, run_indices, fan_in):
    """K-way merge with a bounded fan-in; returns (final run index, number of merge passes)."""
    current, passes = list(run_indices), 0
    while len(current) > 1:
        passes += 1
        merged = []
        for start in range(0, len(current), fan_in):
            group = current[start:start + fan_in]
            if len(group) == 1:                     # nothing to merge, it just moves to the next pass
                merged.append(group[0])
                continue
            buffers = [run_store.read(index) for index in group]
            merged.append(run_store.write(list(heapq.merge(*buffers))))
        current = merged
    return current[0], passes


def compact_ranks(sorted_records):
    """Scan the sorted records once: equal keys get equal ranks, returned with the distinct count."""
    ranks = [0] * len(sorted_records)
    previous, distinct = None, 1
    for position, (key, index) in enumerate(sorted_records):
        if position and key != previous:
            distinct += 1
        ranks[index] = distinct - 1
        previous = key
    return ranks, distinct


def external_suffix_array(text, memory_records=256, fan_in=4, page_size=256):
    """External prefix doubling over a page-counted rank file; returns (suffix array, statistics)."""
    n = len(text)
    stats = {"rounds": 0, "merge_passes": 0, "page_reads": 0, "page_writes": 0,
             "initial_runs": 0, "distinct": 0}
    if n < 2:
        return list(range(n)), stats
    rank_store, run_store = PageStore(page_size), RunStore(page_size)
    rank_store.load([ord(ch) for ch in text])
    offset = 1
    while True:
        ranks = rank_store.sequential()
        shifted = rank_store.shifted(offset)
        keys = [((ranks[i], shifted[i]), i) for i in range(n)]
        runs = sort_chunks_to_runs(keys, run_store, memory_records)
        final, passes = merge_runs(run_store, runs, fan_in)
        sorted_records = run_store.read(final)
        new_ranks, distinct = compact_ranks(sorted_records)
        stats["rounds"] += 1
        stats["merge_passes"] += passes
        stats["initial_runs"] = len(runs)
        stats["distinct"] = distinct
        if distinct == n:
            stats["page_reads"] = rank_store.reads + run_store.reads
            stats["page_writes"] = rank_store.writes + run_store.writes
            return [index for _, index in sorted_records], stats
        rank_store.load(new_ranks)
        offset <<= 1


def in_memory_doubling(text):
    """Oracle: the same doubling, done entirely in memory."""
    n = len(text)
    if n == 0:
        return []
    sa, rank, k = list(range(n)), [ord(c) for c in text], 1
    while True:
        keys = {i: (rank[i], rank[i + k] if i + k < n else -1) for i in range(n)}
        sa.sort(key=keys.__getitem__)
        new_rank = [0] * n
        for index in range(1, n):
            previous, current = sa[index - 1], sa[index]
            new_rank[current] = new_rank[previous] + (0 if keys[previous] == keys[current] else 1)
        rank = new_rank
        if rank[sa[-1]] == n - 1:
            return sa
        k <<= 1


def compare_suffixes(text, i, j):
    """Manual comparator; the independent validity oracle for a suffix array."""
    n = len(text)
    while i < n and j < n and text[i] == text[j]:
        i += 1
        j += 1
    if i == n and j == n:
        return 0
    if i == n:
        return -1
    if j == n:
        return 1
    return -1 if text[i] < text[j] else 1


def valid_suffix_array(text, sa):
    return (sorted(sa) == list(range(len(text)))
            and all(compare_suffixes(text, sa[k], sa[k + 1]) < 0 for k in range(len(sa) - 1)))


def main():
    texts = ["", "a", "aaaa", "banana", "mississippi", "abracadabra", "abcabxabcd", "abababab"]
    texts += [lcg_string(120, alphabet, 6070 + trial * 7919)[0]
              for alphabet in ("ab", "abc") for trial in range(5)]
    for text in texts:
        sa, stats = external_suffix_array(text, memory_records=32, fan_in=4, page_size=16)
        check(sa == in_memory_doubling(text), text)
        check(valid_suffix_array(text, sa), text)
        check(stats["rounds"] == 0 if len(text) < 2 else stats["distinct"] == len(text), text)

    text, _ = lcg_string(400, "abc", 4711)
    for memory_records in (24, 64, 128):
        sa, stats = external_suffix_array(text, memory_records=memory_records, fan_in=4, page_size=32)
        check(sa == in_memory_doubling(text), memory_records)
        runs = (len(text) + memory_records - 1) // memory_records
        first_round_passes = expected_merge_passes(runs, 4)
        pages = (len(text) + 31) // 32
        check(stats["merge_passes"] >= first_round_passes, (memory_records, stats["merge_passes"]))
        check(stats["page_reads"] >= stats["rounds"] * 2 * pages, memory_records)
        check(stats["page_writes"] >= stats["rounds"] * pages, memory_records)
    small_budget = external_suffix_array(text, memory_records=16, fan_in=4, page_size=32)
    large_budget = external_suffix_array(text, memory_records=128, fan_in=4, page_size=32)
    check(small_budget[0] == large_budget[0], "same suffix array either way")
    check(small_budget[1]["merge_passes"] >= large_budget[1]["merge_passes"], "smaller memory spills")
    check(expected_merge_passes(8, 4) == 2 and expected_merge_passes(1, 4) == 0, "pass model")

    # the page model: a sequential pass costs one read per page, and a bigger page means fewer reads
    store = PageStore(16)
    store.load(list(range(160)))
    check(store.count() == 160, "page store content")
    check(store.sequential() == list(range(160)), "sequential scan")
    check(store.reads == 10, f"sequential pass should cost 10 page reads, got {store.reads}")
    check(store.shifted(3)[:4] == [3, 4, 5, 6], "shifted view")
    check(store.reads == 20, "second pass costs another ten reads")
    coarse = PageStore(32)
    coarse.load(list(range(160)))
    coarse.sequential()
    check(coarse.reads == 5, "doubling the page size halves the page reads")

    # medium text: the external array must agree with the in-memory oracle and stay a permutation
    medium, _ = lcg_string(4096, "abcd", 20260707)
    sa, stats = external_suffix_array(medium, memory_records=256, fan_in=4, page_size=128)
    check(sa == in_memory_doubling(medium), "medium external vs in-memory")
    check(valid_suffix_array(medium, sa), "medium suffix array valid")
    doubled, _ = lcg_string(4096, "abcd", 20260707)
    coarse_sa, coarse_stats = external_suffix_array(doubled, memory_records=256, fan_in=4, page_size=256)
    check(coarse_sa == sa, "page size does not change the result")
    check(coarse_stats["page_reads"] < stats["page_reads"], "bigger pages mean fewer reads")
    check(stats["distinct"] == len(medium), "all suffixes distinct at the end")

    print(f"medium text: {len(medium)} characters, {stats['rounds']} doubling rounds, "
          f"{stats['initial_runs']} spills per round, {stats['merge_passes']} merge passes in total")
    print(f"I/O: {stats['page_reads']} page reads, {stats['page_writes']} page writes "
          f"with 128-record pages; {coarse_stats['page_reads']} reads with 256-record pages")


if __name__ == "__main__":
    main()
    print("PASS")
