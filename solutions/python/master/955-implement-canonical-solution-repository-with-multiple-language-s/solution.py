"""Task 955 - Implement canonical solution repository with multiple language solutions and
performance baselines.  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)

Repository model: tasks register one artifact per language; the canonical artifact (the reference
solution every other language is diffed against) is the one from the highest-priority language
present, so the choice never depends on registration order.  Content is hashed after comment
stripping and whitespace normalisation, which makes the hash an identity for "the same program" and
lets the repo flag two distinct artifacts claiming that identity in one language.  Each artifact
carries a median runtime; comparing it with the per-language baseline classifies the build as a
regression, an improvement or stable.

Simplified: artifacts are short code strings, neither compiled nor executed, and runtimes are
declared ticks rather than measurements.  Verified: canonical selection under shuffled registration,
coverage gaps, content-identity normalisation, the duplicate detector, drift classification on both
sides of the tolerance band, and the empty/single-artifact boundaries.
"""
import hashlib
import re

LANGUAGES = ("python", "java", "javascript", "csharp")
DRIFT_REGRESSION, DRIFT_IMPROVEMENT = 1.2, 0.8
COMMENT = re.compile(r"#.*$|//.*$|/\*.*?\*/", re.M)


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def normalise(code):
    """Strip comments and collapse whitespace so formatting cannot change an artifact's identity."""
    return " ".join(COMMENT.sub("", code).split())


def content_hash(code):
    return hashlib.sha256(normalise(code).encode("utf-8")).hexdigest()[:16]


def body(task, language, ticks, extra=""):
    """A tiny but language-flavoured artifact body."""
    comment = {"python": "#", "java": "//", "javascript": "//", "csharp": "//"}[language]
    return (f"{comment} task {task}\n{comment} median {ticks}{extra}\n"
            f"def solve(values):\n    return sum(values)\n")


class Repo:
    """Registration log plus the derived views (canonical, coverage, duplicates, drift)."""

    def __init__(self):
        self.registrations = []   # append-only: one dict per (task, language) submission
        self.baselines = {}

    def register(self, task, language, code, ticks):
        if language not in LANGUAGES:
            raise ValueError(f"unknown language {language!r}")
        entry = {"task": task, "language": language, "code": code,
                 "hash": content_hash(code), "ticks": ticks}
        self.registrations.append(entry)
        return entry

    def set_baseline(self, task, language, ticks):
        self.baselines.setdefault(task, {})[language] = ticks

    def retime(self, task, language, ticks):
        """Re-measure the latest artifact without touching its code."""
        self.artifacts(task)[language]["ticks"] = ticks

    def tasks(self):
        return sorted({entry["task"] for entry in self.registrations})

    def artifacts(self, task):
        """Latest artifact per language for one task."""
        latest = {}
        for entry in self.registrations:
            if entry["task"] == task:
                latest[entry["language"]] = entry
        return latest

    def languages(self, task):
        return sorted(self.artifacts(task))

    def canonical(self, task):
        """Highest-priority language present; (None, None) for a task with no artifact."""
        present = self.artifacts(task)
        for language in LANGUAGES:
            if language in present:
                return language, present[language]
        return None, None

    def coverage(self):
        """Missing languages per task, in priority order; complete tasks are not listed."""
        gaps = {}
        for task in self.tasks():
            missing = [language for language in LANGUAGES if language not in self.artifacts(task)]
            if missing:
                gaps[task] = missing
        return gaps

    def duplicates(self):
        """(task, language) pairs where several submissions share one content hash."""
        groups = {}
        for entry in self.registrations:
            groups.setdefault((entry["task"], entry["language"]), []).append(entry)
        return sorted(key for key, entries in groups.items()
                      if len({entry["hash"] for entry in entries}) < len(entries))

    def drift(self, task):
        """Per-language comparison with the baseline: ratio plus a stable/improved/regressed label."""
        rows = {}
        for language, artifact in sorted(self.artifacts(task).items()):
            baseline = self.baselines.get(task, {}).get(language)
            if baseline is None:
                continue
            ratio = artifact["ticks"] / baseline
            label = ("regressed" if ratio > DRIFT_REGRESSION
                     else "improved" if ratio < DRIFT_IMPROVEMENT else "stable")
            rows[language] = {"ratio": ratio, "label": label}
        return rows

    def canonical_languages(self):
        return {task: self.canonical(task)[0] for task in self.tasks()}

    def report(self):
        """The published view, sorted by task."""
        gaps = self.coverage()
        return {task: {"canonical": self.canonical(task)[0],
                       "hash": self.canonical(task)[1]["hash"],
                       "missing": gaps.get(task, []), "drift": self.drift(task)}
                for task in self.tasks()}


def build_repo(tasks, ticks=100):
    """Register the same problem in every language, with a per-language baseline."""
    repo = Repo()
    for task in tasks:
        for index, language in enumerate(LANGUAGES):
            repo.register(task, language, body(task, language, ticks + index), ticks + index)
            repo.set_baseline(task, language, ticks + index)
    return repo


def main():
    repo = build_repo((955, 956))

    # 1) happy path: a full matrix has no gaps and Python is canonical everywhere.
    check(repo.coverage() == {} and repo.canonical_languages() == {955: "python", 956: "python"}
          and repo.duplicates() == []
          and all(len(repo.artifacts(task)) == 4 for task in (955, 956)),
          f"a full matrix must have no gaps: {repo.coverage()}")

    # 2) canonical selection ignores registration order (oracle: highest-priority present).
    shuffled = Repo()
    for task in (955, 956):
        for language in reversed(LANGUAGES):
            shuffled.register(task, language, body(task, language, 100), 100)
    mixed = Repo()
    mixed.register(7, "javascript", "x = 1", 10)
    mixed.register(7, "java", "int x = 1;", 10)
    check(shuffled.canonical_languages() == repo.canonical_languages()
          and mixed.canonical(7)[0] == "java"
          and mixed.canonical(7)[1]["hash"] == mixed.artifacts(7)["java"]["hash"],
          f"canonical choice must be priority-based: {mixed.canonical_languages()}")

    # 3) coverage gaps are reported exactly, cross-checked against a set-difference oracle.
    partial = Repo()
    partial.register(1, "python", "a", 1)
    partial.register(1, "csharp", "b", 1)
    partial.register(2, "java", "c", 1)
    gaps = partial.coverage()
    oracle = {task: [language for language in LANGUAGES if language not in langs]
              for task, langs in ((1, ("python", "csharp")), (2, ("java",)))}
    check(gaps == {1: ["java", "javascript"], 2: ["python", "javascript", "csharp"]}
          and gaps == oracle,
          f"coverage gaps wrong: {gaps}")

    # 4) duplicate detector: two submissions in one language with the same normalised content,
    #    while a genuinely different program in the same language is not flagged.
    conflicting = Repo()
    conflicting.register(3, "python", "def solve(v):\n    return sum(v)\n", 10)
    conflicting.register(3, "python", "def solve(v):  # tuned\n    return sum(v)\n", 9)
    conflicting.register(3, "java", "class S {}", 10)
    check(conflicting.duplicates() == [(3, "python")]
          and len({entry["hash"] for entry in conflicting.registrations}) == 2,
          f"the detector must flag only the python pair: {conflicting.duplicates()}")

    # 5) content identity is normalisation based; a real algorithm change must not collide.
    check(content_hash(body(955, "python", 100))
          == content_hash("  def solve(values):  # inline\n\n    return   sum(values)\n")
          and content_hash(body(955, "python", 100))
          != content_hash(body(955, "python", 100).replace("sum", "sorted")),
          "comments and whitespace must not change an artifact's identity")

    # 6) drift classification on both sides of the band, including the exact boundary ratio.
    drifted = build_repo((955,))
    for language in LANGUAGES:
        drifted.set_baseline(955, language, 100)
    drifted.retime(955, "python", 120)       # ratio 1.20 -> still stable
    drifted.retime(955, "java", 121)         # ratio 1.21 -> regressed
    drifted.retime(955, "javascript", 60)    # ratio 0.60 -> improved
    rows = drifted.drift(955)
    labels = {language: row["label"] for language, row in rows.items()}
    check(labels == {"python": "stable", "java": "regressed", "javascript": "improved",
                     "csharp": "stable"} and rows["java"]["ratio"] == 1.21
          and rows["python"]["ratio"] == 1.2,
          f"drift labels wrong: {labels}")

    # 7) boundaries: an empty repo publishes nothing; a single artifact leaves three gaps.
    empty = Repo()
    single = Repo()
    single.register(42, "csharp", "class S {}", 5)
    single.set_baseline(42, "csharp", 5)
    check(empty.canonical(1) == (None, None) and empty.coverage() == {} and empty.report() == {}
          and empty.tasks() == [] and single.canonical(42)[0] == "csharp"
          and single.coverage() == {42: ["python", "java", "javascript"]}
          and single.drift(42)["csharp"]["label"] == "stable",
          f"empty/single boundary wrong: {empty.report()} / {single.coverage()}")

    # 8) the published report is complete and consistent, and re-computing it is deterministic.
    published = repo.report()
    consistent = all(
        published[task]["canonical"] in repo.languages(task)
        and set(published[task]["missing"]) | set(repo.languages(task)) == set(LANGUAGES)
        and published[task]["hash"] == repo.canonical(task)[1]["hash"]
        for task in published)
    check(consistent and published == repo.report() and sorted(published) == [955, 956]
          and len(repo.registrations) == 8,
          f"the report must be complete and reproducible: {sorted(published)}")

    # 9) invariant after mutation: re-registering a non-canonical artifact changes that language's
    #    drift but must never move the canonical pointer, and baselines stay untouched.
    before = {task: dict(values) for task, values in repo.baselines.items()}
    repo.register(955, "csharp", "class Faster { /* tuned */ }", 10)
    check(repo.baselines == before and repo.canonical_languages()[955] == "python"
          and repo.artifacts(955)["csharp"]["ticks"] == 10
          and repo.drift(955)["csharp"]["label"] == "improved" and repo.coverage() == {},
          "re-registering a non-canonical artifact must not move the canonical pointer")

    print(f"tasks={repo.tasks()} canonical={repo.canonical_languages()} gaps={repo.coverage()} "
          f"duplicates={conflicting.duplicates()}")
    print(f"drift={ {lang: row['label'] for lang, row in rows.items()} } "
          f"hashes={[repo.artifacts(955)[lang]['hash'][:6] for lang in LANGUAGES]}")
    print("PASS")


if __name__ == "__main__":
    main()
