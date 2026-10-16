"""Task 985 - Implement contribution recognition system with verified citations for authors.
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

A recognition engine over credited contributions.  Every contribution carries a citation
whose SHA-256 digest must match bytes a maintainer really archived and whose source must be
an allowlisted HTTPS host; only verified contributions reach the scoreboard, so a published
score cannot be inflated by an unverifiable claim.  Points are ``weight[kind] * rating`` for
a 1..5 maintainer grade, contributions de-duplicate by (author, subject, digest), and equal
points share a competition rank (1, 2, 2, 4).

Simplified versus production: the artifact store is an in-process digest -> bytes map instead
of object storage, and the host allowlist is a fixed tuple.  Verified here: digest/host/rating
rejection, duplicate suppression, that the reported total always equals the sum over verified
contributions, and that ranking agrees with an independent "1 + strictly better authors"
computation, including ties.
"""
import hashlib

WEIGHTS = {"code": 5, "review": 3, "design": 4, "docs": 2}
BADGES = ((40, "platinum"), (20, "gold"), (8, "silver"), (1, "bronze"))
ALLOWED_HOSTS = ("github.com", "gitlab.com", "cracking-coding.dev")


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def host_allowed(url):
    """Only https citations on an allowlisted host count as evidence of authorship."""
    if not url.startswith("https://"):
        return False
    host = url[len("https://"):].split("/", 1)[0].split(":", 1)[0]
    return host.lower() in ALLOWED_HOSTS


def badge_for(points):
    for threshold, name in BADGES:
        if points >= threshold:
            return name
    return "none"


class Contribution:
    """One credited unit of work plus the citation that is supposed to prove it."""

    def __init__(self, author, kind, subject, rating, source, artifact):
        self.author, self.kind, self.subject = author, kind, subject
        self.rating, self.source, self.artifact = rating, source, artifact
        self.digest = sha256_text(artifact)

    @property
    def points(self):
        return WEIGHTS[self.kind] * self.rating

    @property
    def key(self):
        return (self.author, self.subject, self.digest)


class CitationRegistry:
    """digest -> archived text for artifacts that were actually merged."""

    def __init__(self):
        self.archived = {}

    def archive(self, text):
        digest = sha256_text(text)
        self.archived[digest] = text
        return digest

    def verify(self, contribution):
        if contribution.kind not in WEIGHTS:
            return False, "unknown contribution kind"
        if not 1 <= contribution.rating <= 5:
            return False, "rating outside 1..5"
        if not host_allowed(contribution.source):
            return False, "citation host not allowlisted or not https"
        stored = self.archived.get(contribution.digest)
        if stored is None:
            return False, "artifact digest was never archived"
        if stored != contribution.artifact:
            return False, "archived bytes differ from the quoted artifact"
        return True, "verified"


class RecognitionSystem:
    def __init__(self):
        self.registry = CitationRegistry()
        self.credit = {}
        self.rejected = []

    def submit(self, contribution):
        ok, reason = self.registry.verify(contribution)
        if not ok:
            self.rejected.append((contribution.author, contribution.subject, reason))
            return False
        self.credit.setdefault(contribution.key, contribution)
        return True

    def total_points(self):
        return sum(c.points for c in self.credit.values())

    def scoreboard(self):
        totals = {}
        for contribution in self.credit.values():
            totals[contribution.author] = totals.get(contribution.author, 0) + contribution.points
        ordered = sorted(totals.items(), key=lambda pair: (-pair[1], pair[0]))
        rows, rank, previous = [], 0, None
        for position, (author, points) in enumerate(ordered, start=1):
            if points != previous:
                rank, previous = position, points
            rows.append({"rank": rank, "author": author, "points": points,
                         "badge": badge_for(points)})
        return rows


def splitmix64(seed, count):
    """Deterministic pseudo-random stream: fixed seed, no stdlib random, no wall clock."""
    state, out = seed & ((1 << 64) - 1), []
    for _ in range(count):
        state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
        out.append(z ^ (z >> 31))
    return out


def main():
    patches = {"bubble-sort": "def bubble(a):\n    return sorted(a)\n",
               "lru-cache": "class LRU: pass\n",
               "editorial-42": "Explaining amortised O(1) append.\n",
               "review-77": "LGTM, but the loop bound is off by one.\n"}
    registry = CitationRegistry()
    for text in patches.values():
        registry.archive(text)
    system = RecognitionSystem()
    system.registry = registry

    def credit(target, author, kind, subject, rating, host="github.com"):
        return target.submit(Contribution(author, kind, subject, rating,
                                          "https://%s/%s/%s" % (host, author, subject),
                                          patches[subject]))

    # Happy path: mixed kinds and grades, checked against hand-computed points.
    check(credit(system, "ada", "code", "lru-cache", 5), "ada's code citation must verify")
    check(credit(system, "ada", "review", "review-77", 4), "ada's review citation must verify")
    check(credit(system, "grace", "design", "bubble-sort", 5, "gitlab.com"), "design must verify")
    check(credit(system, "linus", "docs", "editorial-42", 3, "cracking-coding.dev"), "docs")
    expected = {"ada": 5 * 5 + 3 * 4, "grace": 4 * 5, "linus": 2 * 3}  # weight * rating oracle
    board = system.scoreboard()
    check([row["author"] for row in board] == ["ada", "grace", "linus"],
          "scoreboard must be ordered by descending points")
    check({row["author"]: row["points"] for row in board} == expected,
          "points differ from the independent weight*rating oracle")
    check(system.total_points() == sum(expected.values()), "total disagrees with the oracle")
    check(board[0]["badge"] == "gold" and board[-1]["badge"] == "bronze",
          "badges must follow the point totals")

    # Rank semantics: ties share a rank and the following rank skips (1, 1, 3, 4).
    tie = RecognitionSystem()
    for index, (author, rating) in enumerate([("a", 3), ("b", 3), ("c", 2), ("d", 1)]):
        text = "tie-artifact-%d" % index
        tie.registry.archive(text)
        tie.submit(Contribution(author, "docs", "tie-%d" % index, rating,
                                "https://github.com/x/%d" % index, text))
    tie_board = tie.scoreboard()
    check([row["points"] for row in tie_board] == [6, 6, 4, 2], "unexpected tie point vector")
    check([row["rank"] for row in tie_board] == [1, 1, 3, 4], "competition ranks must skip")
    # Independent rank formula: rank = 1 + number of authors that are strictly better.
    scores = {"a": 6, "b": 6, "c": 4, "d": 2}
    oracle = {name: 1 + sum(1 for other in scores.values() if other > own)
              for name, own in scores.items()}
    check(all(row["rank"] == oracle[row["author"]] for row in tie_board),
          "ranks disagree with the 1 + strictly-greater-count oracle")

    # Boundary: the badge threshold is inclusive (8 -> silver) and 1 below is bronze.
    check(badge_for(8) == "silver" and badge_for(7) == "bronze",
          "badge thresholds must be inclusive at the boundary")
    check(badge_for(40) == "platinum" and badge_for(0) == "none", "badge edges are wrong")

    # Rejection: tampered artifact, plain http, non-allowlisted host, bad rating.
    for source, artifact, rating in (
            ("https://github.com/mallory/x", patches["lru-cache"] + "# extra\n", 5),
            ("http://github.com/mallory/x", patches["lru-cache"], 5),
            ("https://example.com/mallory/x", patches["lru-cache"], 5),
            ("https://github.com/mallory/x", patches["lru-cache"], 6)):
        check(not system.submit(Contribution("mallory", "code", "lru-cache", rating,
                                            source, artifact)),
              "an unverifiable citation must be rejected: %s" % source)
    check(len(system.rejected) == 4, "every rejected claim must be logged")
    check(system.total_points() == sum(expected.values()) and len(system.credit) == 4,
          "rejected claims must not change the verified pool")

    # Duplicate suppression: an identical (author, subject, digest) triple counts once.
    again = Contribution("ada", "code", "lru-cache", 5, "https://github.com/ada/lru",
                         patches["lru-cache"])
    before = system.total_points()
    check(system.submit(again), "an identical re-submission is still a verified citation")
    check(system.total_points() == before and len(system.credit) == 4,
          "the duplicate must not add points a second time")

    # Property + oracle over a generated field: group-by aggregation of the same stream.
    kinds = ("code", "review", "design", "docs")
    generated, grouped = RecognitionSystem(), {}
    for index, value in enumerate(splitmix64(2026, 32)):
        author = "author-%d" % (value % 7)
        kind, rating = kinds[(value >> 16) % 4], 1 + (value >> 8) % 5
        text = "generated-artifact-%d" % index
        generated.registry.archive(text)
        check(generated.submit(Contribution(author, kind, "gen-%d" % index, rating,
                                            "https://github.com/gen/%d" % index, text)),
              "generated citation must verify")
        grouped.setdefault(author, {}).setdefault(kind, []).append(rating)
        # Invariant re-checked after every mutation, against an independent accumulator.
        table = sum(sum(cells[k]) * WEIGHTS[k] for cells in grouped.values() for k in cells)
        check(generated.total_points() == table, "running total drifted at step %d" % index)
    aggregate = {author: sum(sum(ratings) * WEIGHTS[k] for k, ratings in cells.items())
                 for author, cells in grouped.items()}
    check({row["author"]: row["points"] for row in generated.scoreboard()} == aggregate,
          "scoreboard disagrees with the group-by aggregation oracle")
    check(generated.scoreboard()[0]["points"] == max(aggregate.values()),
          "the leader must hold the maximum recomputed score")

    # Boundary: an empty system yields no rows, no rank and a zero total.
    empty = RecognitionSystem()
    check(empty.scoreboard() == [] and empty.total_points() == 0 and not empty.rejected,
          "an empty recognition system must produce an empty scoreboard")

    print("verified=%d rejected=%d total_points=%d leader=%s"
          % (len(system.credit), len(system.rejected), system.total_points(), board[0]["author"]))
    print("PASS")


if __name__ == "__main__":
    main()
