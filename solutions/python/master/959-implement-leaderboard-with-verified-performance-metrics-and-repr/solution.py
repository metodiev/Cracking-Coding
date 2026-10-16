"""Task 959 - Implement leaderboard with verified performance metrics and reproducible runs.
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

Leaderboard model: a submission is a run (user, task, language, ticks, content digest, machine).  A
run is *verified* only when at least two independent machines reproduce the same digest and their
ticks agree inside a spread tolerance; the user's verified metric is the best tick in that group.  A
group whose spread exceeds the tolerance is flaky and is not ranked at all, and a tick below the
physical floor is rejected outright.  Unverified claims stay visible in their own list, so a fast
unverified time can never outrank a slower verified one.

Simplified: ticks are declared integers rather than measurements, digests are opaque strings and
"machines" are labels.  Verified: verification of reproducible groups, best-of-verified selection,
unverified and rejected buckets, flaky-group exclusion, deterministic tie-breaking, bucket
conservation, and the empty/single-run boundaries.
"""
FLOOR_TICKS = 1
SPREAD_TOLERANCE = 1.05


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Board:
    """Run store plus the derived verification, ranking and reporting views."""

    def __init__(self, floor=FLOOR_TICKS, spread_tolerance=SPREAD_TOLERANCE):
        self.runs = []
        self.floor = floor
        self.spread_tolerance = spread_tolerance

    def submit(self, user, task, language, ticks, digest, machine):
        entry = {"index": len(self.runs), "user": user, "task": task, "language": language,
                 "ticks": ticks, "digest": digest, "machine": machine}
        self.runs.append(entry)
        return entry

    def entries(self, task, language):
        return [entry for entry in self.runs
                if entry["task"] == task and entry["language"] == language]

    def credited(self, task, language):
        """Runs at or above the physical floor; anything faster is rejected, never verified."""
        return [entry for entry in self.entries(task, language) if entry["ticks"] >= self.floor]

    def groups(self, task, language):
        """Digest -> credited runs, keeping submission order inside each group."""
        grouped = {}
        for entry in self.credited(task, language):
            grouped.setdefault(entry["digest"], []).append(entry)
        return grouped

    def verify(self, task, language):
        """Split the runs into verified groups, flaky groups and unverified singletons."""
        verified, flaky, unverified = [], [], []
        for digest in sorted(self.groups(task, language)):
            runs = self.groups(task, language)[digest]
            machines = {entry["machine"] for entry in runs}
            if len(machines) < 2:
                unverified.extend(runs)
                continue
            ticks = [entry["ticks"] for entry in runs]
            spread = max(ticks) / min(ticks) if min(ticks) > 0 else float("inf")
            row = {"digest": digest, "runs": runs, "metric": min(ticks), "spread": spread,
                   "user": min(runs, key=lambda entry: (entry["ticks"], entry["index"]))["user"],
                   "first_index": min(entry["index"] for entry in runs)}
            (flaky if spread > self.spread_tolerance else verified).append(row)
        return {"verified": sorted(verified, key=lambda row: (row["metric"], row["first_index"])),
                "flaky": sorted(flaky, key=lambda row: row["digest"]),
                "unverified": unverified}

    def rejected(self, task, language):
        return [entry for entry in self.entries(task, language) if entry["ticks"] < self.floor]

    def ranked(self, task, language):
        """Verified rows in rank order; ties break on first submission index then user name."""
        rows = self.verify(task, language)["verified"]
        ordered = sorted(rows, key=lambda row: (row["metric"], row["first_index"], row["user"]))
        return [{"rank": position + 1, "user": row["user"], "ticks": row["metric"],
                 "digest": row["digest"], "spread": row["spread"]}
                for position, row in enumerate(ordered)]

    def keys(self):
        return sorted({(entry["task"], entry["language"]) for entry in self.runs})

    def report(self):
        return {key: {"ranked": self.ranked(*key),
                      "flaky": [row["digest"] for row in self.verify(*key)["flaky"]],
                      "unverified": sorted(entry["user"] for entry in
                                           self.verify(*key)["unverified"]),
                      "rejected": sorted(entry["user"] for entry in self.rejected(*key))}
                for key in self.keys()}

    def bucket(self, task, language):
        """Every run, exactly once, in the bucket it landed in."""
        parts = self.verify(task, language)
        names = [entry["user"] for row in parts["verified"] for entry in row["runs"]]
        names += [entry["user"] for row in parts["flaky"] for entry in row["runs"]]
        names += [entry["user"] for entry in parts["unverified"]]
        names += [entry["user"] for entry in self.rejected(task, language)]
        return names


def seeded_board():
    """A board with reproducible groups, a flaky group, unverified claims and a rejection."""
    board = Board()
    board.submit("dana", 959, "python", 120, "sha-A", "m1")     # verified: 120 / 121
    board.submit("dana", 959, "python", 121, "sha-A", "m2")
    board.submit("erin", 959, "python", 999, "sha-B", "m3")     # fast but never reproduced
    board.submit("frank", 959, "python", 130, "sha-C", "m4")    # flaky: 130 vs 205
    board.submit("frank", 959, "python", 205, "sha-C", "m5")
    board.submit("gina", 959, "python", 118, "sha-D", "m6")     # verified and fast
    board.submit("gina", 959, "python", 119, "sha-D", "m7")
    board.submit("hugo", 959, "python", 0, "sha-E", "m8")       # below the physical floor
    return board


def main():
    board = seeded_board()

    # 1) happy path (the point of verification): a slower verified time outranks a faster claim
    #    that was never reproduced.
    ranking = board.ranked(959, "python")
    check([row["user"] for row in ranking] == ["gina", "dana"]
          and [row["ticks"] for row in ranking] == [118, 120]
          and all(row["user"] != "erin" for row in ranking),
          f"verified runs must be ranked without the unverified claim: {ranking}")

    # 2) the verified metric is the best tick in the reproducible group, and the spread is reported.
    verify = board.verify(959, "python")
    dana = next(row for row in verify["verified"] if row["user"] == "dana")
    check(dana["metric"] == 120 and len(dana["runs"]) == 2
          and abs(dana["spread"] - 121 / 120) < 1e-12,
          f"best-of-verified wrong: {dana['metric']} spread {dana['spread']}")

    # 3) unverified claims stay visible but unranked; flaky groups are excluded with their digest
    #    reported so a reviewer can look at the machine pair.
    unverified_users = sorted(entry["user"] for entry in verify["unverified"])
    check(unverified_users == ["erin"] and [row["digest"] for row in verify["flaky"]] == ["sha-C"]
          and verify["flaky"][0]["spread"] > SPREAD_TOLERANCE,
          f"unverified/flaky split wrong: {unverified_users} {verify['flaky']}")

    # 4) an independent oracle for verification: a group is verified iff it spans >= 2 machines and
    #    its spread is inside the tolerance; recomputed from the raw runs.
    oracle = []
    for digest, runs in board.groups(959, "python").items():
        machines = {entry["machine"] for entry in runs}
        ticks = [entry["ticks"] for entry in runs]
        if len(machines) >= 2 and max(ticks) / min(ticks) <= SPREAD_TOLERANCE:
            oracle.append(digest)
    check(sorted(oracle) == sorted(row["digest"] for row in verify["verified"])
          and len(oracle) == 2,
          f"verification disagrees with the oracle: {oracle}")

    # 5) tie-breaking is deterministic: equal metrics fall back to the first submission index and
    #    then the user name; the documented key is checked directly on synthetic rows, and swapping
    #    the machines inside each group leaves the ranking untouched.
    tie = Board()
    tie.submit("zoe", 1, "python", 100, "d1", "m1")
    tie.submit("zoe", 1, "python", 100, "d1", "m2")
    tie.submit("amy", 1, "python", 100, "d2", "m3")
    tie.submit("amy", 1, "python", 100, "d2", "m4")
    swapped = Board()
    swapped.submit("zoe", 1, "python", 100, "d1", "m2")
    swapped.submit("zoe", 1, "python", 100, "d1", "m1")
    swapped.submit("amy", 1, "python", 100, "d2", "m4")
    swapped.submit("amy", 1, "python", 100, "d2", "m3")
    names = lambda rows: [row["user"] for row in rows]
    synthetic = sorted([{"metric": 100, "first_index": 5, "user": "zoe"},
                        {"metric": 100, "first_index": 5, "user": "amy"}],
                       key=lambda row: (row["metric"], row["first_index"], row["user"]))
    check(names(tie.ranked(1, "python")) == ["zoe", "amy"]
          and names(swapped.ranked(1, "python")) == names(tie.ranked(1, "python"))
          and [row["rank"] for row in tie.ranked(1, "python")] == [1, 2]
          and [row["user"] for row in synthetic] == ["amy", "zoe"],
          f"tie-breaking must be deterministic: {tie.ranked(1, 'python')}")

    # 6) bucket conservation: every submitted run lands in exactly one bucket.
    buckets = board.bucket(959, "python")
    check(sorted(buckets) == sorted(entry["user"] for entry in board.entries(959, "python"))
          and len(buckets) == 8,
          f"runs must be accounted for exactly once: {buckets}")

    # 7) boundaries: an empty board has no keys and no ranking, and a lone run can never be verified
    #    because reproducibility requires a second machine.
    empty, single = Board(), Board()
    single.submit("solo", 959, "python", 50, "sha-S", "m1")
    check(empty.keys() == [] and empty.report() == {}
          and single.ranked(959, "python") == []
          and len(single.verify(959, "python")["unverified"]) == 1
          and single.verify(959, "python")["verified"] == [],
          f"empty/single boundary wrong: {single.verify(959, 'python')}")

    # 8) rejection versus unverified: a tick below the floor is rejected with a reason, and the
    #    rejection is a property of the metric, not of reproducibility.
    check([entry["user"] for entry in board.rejected(959, "python")] == ["hugo"]
          and "hugo" in board.report()[(959, "python")]["rejected"]
          and board.report()[(959, "python")]["ranked"][0]["ticks"] == 118
          and [row["ticks"] for row in board.ranked(959, "python")]
          == sorted(row["ticks"] for row in board.ranked(959, "python")),
          f"rejection handling wrong: {board.report()[(959, 'python')]['rejected']}")

    # 9) determinism: recomputing the report twice, and after an unrelated submission in another
    #    language, leaves this language's ranking unchanged.
    before = board.report()[(959, "python")]
    board.submit("dana", 959, "java", 500, "sha-J", "m9")
    check(board.report()[(959, "python")] == before
          and board.report() == board.report()
          and [entry["user"] for entry in board.verify(959, "java")["unverified"]] == ["dana"],
          "the leaderboard must be deterministic and per-language isolated")

    print(f"ranked={[(row['rank'], row['user'], row['ticks']) for row in ranking]} "
          f"unverified={unverified_users} flaky={[row['digest'] for row in verify['flaky']]}")
    print(f"rejected={board.report()[(959, 'python')]['rejected']} "
          f"verified_groups={len(verify['verified'])} spread={verify['flaky'][0]['spread']:.3f}")
    print("PASS")


if __name__ == "__main__":
    main()
