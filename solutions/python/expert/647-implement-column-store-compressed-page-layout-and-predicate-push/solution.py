"""Task 647 - Implement column-store compressed page layout and predicate pushdown simulation.  (Tier: Expert, Ch 14 — Databases)

A column page layout for a column store, plus predicate pushdown into the compressed data.
A column is dictionary-encoded (one global dictionary, codes assigned in sorted value
order) and sliced into fixed-size pages; each page stores its values as *runs*
(code, length) - a page-local run-length encoding - together with a zone map (min/max of
its non-NULL values).  Compression is the three classic column-store tricks: dictionary,
RLE and zone maps; `bit_width = ceil(log2(distinct))` is the smallest code width that fits.

Predicate pushdown has two levels.  A page whose zone map cannot contain a match is skipped
without being decoded at all; inside a page a whole run of equal values is tested once, and
only runs that pass are materialized.  Both levels can only *reduce* work - never lose a
row - and the checks pin exactly that: the pushed-down scan must return the same row ids as
a full decode followed by a filter, on hand-built and random columns and predicates.

Simplified: pages are Python objects rather than byte buffers, and predicates are
range or match-all objects, not a full expression compiler.
"""

import math

NULL = None


def sort_key(value):
    """NULL sorts last so that the dictionary order is deterministic."""
    return (1, 0) if value is NULL else (0, value)


class RangePredicate:
    def __init__(self, low, high):
        self.low, self.high = low, high

    def matches(self, value):
        return value is not NULL and self.low <= value <= self.high

    def can_prune(self, zone):
        if zone is None:
            return True                              # a NULL-only page cannot match
        low, high = zone
        return high < self.low or low > self.high


class MatchAll:
    """Pushdown for nothing: shows how much the zone map and RLE tests actually save."""

    def matches(self, value):
        return True

    def can_prune(self, zone):
        return False


class EncodedColumn:
    def __init__(self, dictionary, pages, page_size, total):
        self.dictionary = dictionary
        self.pages = pages
        self.page_size = page_size
        self.total = total

    def runs(self):
        return sum(len(page["runs"]) for page in self.pages)

    def size(self):
        """Compressed size in slots: one slot per run plus one per dictionary entry."""
        return self.runs() + len(self.dictionary)

    def bit_width(self):
        return max(1, math.ceil(math.log2(max(1, len(self.dictionary)))))


def encode_column(values, page_size):
    distinct = sorted({value for value in values}, key=sort_key)
    code_of = {value: code for code, value in enumerate(distinct)}
    pages = []
    for start in range(0, len(values), page_size):
        chunk = values[start:start + page_size]
        runs = []
        for value in chunk:
            code = code_of[value]
            if runs and runs[-1][0] == code:
                runs[-1][1] += 1
            else:
                runs.append([code, 1])
        present = [value for value in chunk if value is not NULL]
        zone = (min(present), max(present)) if present else None
        pages.append({"start": start, "runs": runs, "zone": zone})
    return EncodedColumn(distinct, pages, page_size, len(values))


def decode_column(encoded):
    values = []
    for page in encoded.pages:
        for code, length in page["runs"]:
            values.extend([encoded.dictionary[code]] * length)
    return values


def select(encoded, predicate):
    """Scan the compressed column; returns (matching row ids, statistics)."""
    matches = []
    stats = {"values_decoded": 0, "pages_read": 0, "pages_skipped": 0,
             "runs_tested": 0, "runs_filtered": 0}
    for page in encoded.pages:
        if predicate.can_prune(page["zone"]):
            stats["pages_skipped"] += 1
            continue
        stats["pages_read"] += 1
        row = page["start"]
        for code, length in page["runs"]:
            stats["runs_tested"] += 1
            value = encoded.dictionary[code]
            if predicate.matches(value):              # one test for the whole run
                stats["values_decoded"] += length
                matches.extend(range(row, row + length))
            else:
                stats["runs_filtered"] += 1
            row += length
    return matches, stats


def reference_select(values, predicate):
    return [index for index, value in enumerate(values) if predicate.matches(value)]


def independent_runs(values, page_size):
    """Oracle for the run count: maximal equal-value groups inside each page."""
    total = 0
    for start in range(0, len(values), page_size):
        chunk = values[start:start + page_size]
        total += 1 + sum(1 for i in range(1, len(chunk)) if chunk[i] != chunk[i - 1])
    return total


def lcg(state):
    return (state * 1103515245 + 12345) % (2 ** 31)


def random_column(count, cardinality, seed, null_every=0):
    values, state = [], seed
    for _ in range(count):
        state = lcg(state)
        values.append(NULL if (null_every and state % null_every == 0) else state % cardinality)
    return values, state


def main():
    # round trips ------------------------------------------------------------------------
    for values, page_size in (([], 4), ([5], 4), ([7] * 9, 4), (list(range(10)), 3),
                              ([1, NULL, 2, NULL, NULL], 2), ([3, 3, 4, 3, 3], 3),
                              ([i // 5 for i in range(23)], 5)):
        encoded = encode_column(values, page_size)
        check(decode_column(encoded) == values, "round trip failed for %r" % (values,))
        check(len(encoded.dictionary) == len(set(values)), "dictionary is the distinct set")
        check(encoded.runs() == independent_runs(values, page_size),
              "run count matches the independent run counter")
        check(all(len(page["runs"]) <= page_size for page in encoded.pages),
              "no page stores more runs than it has slots")
        check(all(code < 2 ** encoded.bit_width() for page in encoded.pages
                  for code, _ in page["runs"]), "codes fit the computed bit width")
        check(len(encoded.pages) == math.ceil(len(values) / page_size) or not values,
              "page count is ceil(n / page_size)")

    # the empty column is a real boundary case -------------------------------------------
    empty = encode_column([], 8)
    check(empty.pages == [] and empty.dictionary == [] and empty.size() == 0,
          "an empty column has no pages and no dictionary")
    check(select(empty, RangePredicate(0, 100)) == ([], {
        "values_decoded": 0, "pages_read": 0, "pages_skipped": 0,
        "runs_tested": 0, "runs_filtered": 0}), "an empty column matches nothing")

    # compression: RLE plus a small dictionary beats the raw representation --------------
    clustered = [i // 25 for i in range(500)]                 # 20 long runs
    scattered = [i % 250 for i in range(500)]                 # few repeats
    encoded_clustered = encode_column(clustered, 50)
    encoded_scattered = encode_column(scattered, 50)
    check(len(encoded_clustered.dictionary) == 20, "the clustered column has 20 distinct values")
    check(encoded_clustered.runs() == 20, "500 clustered values collapse into 20 runs")
    check(encoded_clustered.size() < 100 < 500, "the clustered column compresses %d -> %d"
          % (500, encoded_clustered.size()))
    check(encoded_clustered.size() < encoded_scattered.size(),
          "low cardinality compresses better: %d < %d"
          % (encoded_clustered.size(), encoded_scattered.size()))
    check(encoded_scattered.runs() > 400, "the scattered column barely compresses")
    check(encoded_clustered.bit_width() == 5 and encoded_scattered.bit_width() == 8,
          "bit width follows ceil(log2(distinct))")

    # pushdown must never lose a match ---------------------------------------------------
    column = [i // 10 for i in range(400)]                   # 40 runs of 10 identical values
    encoded = encode_column(column, 40)
    for low, high in ((0, 0), (5, 5), (10, 19), (30, 39), (39, 39), (0, 39), (100, 200),
                      (-5, -1), (20, 25)):
        predicate = RangePredicate(low, high)
        matches, stats = select(encoded, predicate)
        check(matches == reference_select(column, predicate),
              "pushdown disagrees with the reference for [%d, %d]" % (low, high))
        check(stats["values_decoded"] <= encoded.total, "never decodes more than the column")
        check(len(matches) <= len(column), "never returns more rows than exist")

    # pushdown really reduces work -------------------------------------------------------
    selective = RangePredicate(30, 31)
    matches, sel_stats = select(encoded, selective)
    check(matches == reference_select(column, selective) and len(matches) == 20,
          "the selective predicate matches two runs")
    check(sel_stats["values_decoded"] == 20, "only the two matching runs were decoded")
    check(sel_stats["runs_filtered"] == 2,
          "the two non-matching runs of the surviving page were rejected as runs")
    check(sel_stats["pages_skipped"] == 9, "zone maps skipped 9 of the 10 pages")
    check(sel_stats["pages_read"] + sel_stats["pages_skipped"] == len(encoded.pages),
          "every page was either read or skipped")

    all_matches, all_stats = select(encoded, MatchAll())
    check(all_matches == list(range(len(column))), "a match-all predicate returns every row")
    check(all_stats["values_decoded"] == len(column) and all_stats["pages_skipped"] == 0,
          "without pushdown every value is decoded and no page is skipped")
    check(sel_stats["values_decoded"] < all_stats["values_decoded"],
          "pushdown decoded %d values instead of %d"
          % (sel_stats["values_decoded"], all_stats["values_decoded"]))
    check(sel_stats["pages_read"] < all_stats["pages_read"],
          "pushdown read %d pages instead of %d"
          % (sel_stats["pages_read"], all_stats["pages_read"]))

    # NULLs are stored, never matched ----------------------------------------------------
    nullable = [1, NULL, 2, NULL, 3, 3, NULL, 4]
    encoded_nulls = encode_column(nullable, 3)
    check(NULL in encoded_nulls.dictionary, "NULL has its own dictionary entry")
    check(decode_column(encoded_nulls) == nullable, "NULLs survive the round trip")
    null_matches, null_stats = select(encoded_nulls, RangePredicate(0, 10))
    check(null_matches == reference_select(nullable, RangePredicate(0, 10)),
          "a range predicate ignores NULLs")
    check(null_stats["values_decoded"] == 5, "the run of NULLs was rejected as a run")

    # property test: random columns, random predicates ------------------------------------
    state = 424242
    for trial in range(25):
        state = lcg(state)
        count = state % 60
        cardinality = 1 + (state >> 5) % 8
        values, state = random_column(count, cardinality, state, null_every=5)
        page_size = 1 + (state >> 3) % 7
        column_encoded = encode_column(values, page_size)
        check(decode_column(column_encoded) == values, "random round trip %d" % trial)
        low = (state >> 7) % cardinality
        high = low + (state >> 11) % cardinality
        predicate = RangePredicate(low, high)
        matches, trial_stats = select(column_encoded, predicate)
        check(matches == reference_select(values, predicate),
              "random pushdown mismatch at trial %d ([%d, %d])" % (trial, low, high))
        check(trial_stats["values_decoded"] <= len(values), "decoded count stays bounded")
        check(0 < trial_stats["runs_tested"] <= column_encoded.runs(),
              "run tests stay within the stored runs")
        check(trial_stats["pages_read"] + trial_stats["pages_skipped"]
              == len(column_encoded.pages),
              "every page was either read or skipped at trial %d" % trial)

    # a single-value page layout is the degenerate case -----------------------------------
    unit_pages = encode_column([2, 2, 3], 1)
    check(all(len(page["runs"]) == 1 for page in unit_pages.pages),
          "page_size 1 gives one run per page")
    check(decode_column(unit_pages) == [2, 2, 3], "unit pages still decode correctly")
    print("[647] column store: 500 clustered values -> %d slots, selective scan decoded "
          "%d of %d values across %d of %d pages"
          % (encoded_clustered.size(), sel_stats["values_decoded"], encoded.total,
             sel_stats["pages_read"], len(encoded.pages)))


def check(ok, message):
    if not ok:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
    print("PASS")
