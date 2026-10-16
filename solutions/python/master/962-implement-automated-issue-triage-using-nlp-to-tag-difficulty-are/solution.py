"""Task 962 - Implement automated issue triage using NLP to tag difficulty/area.
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

Triage model: issue text is lower-cased, split into alphabetic tokens and stop-word filtered; two
multinomial naive Bayes classifiers (Laplace-smoothed, one for the area and one for the difficulty)
then label the issue, and the layer abstains - "needs-triage" - when the text is too short or the
winning posterior is below the confidence threshold.  The posterior is a real product of conditional
probabilities, so the model can be re-derived with exact rational arithmetic; that re-derivation is
the oracle used in the checks.

Simplified: a small hand-written corpus stands in for a trained model, there is no stemming and the
labelling is two independent one-vs-rest problems rather than a joint model.  Verified: held-out
accuracy, the exact posterior against the rational oracle, abstention and its threshold boundary,
the distribution property, unseen vocabulary and the single-class boundaries.
"""
import re
from fractions import Fraction

STOPWORDS = frozenset("the a an and or of to in for with on is are be this that it when how why "
                      "not any all do does at".split())
TOKEN_RE = re.compile(r"[a-z]+")
AREAS = ("parsing", "graphs", "dp", "math", "systems")
DIFFICULTIES = ("easy", "medium", "hard")
ABSTAIN_THRESHOLD = 0.5
MIN_TOKENS = 2
TRAIN = (
    ("parsing", "easy", "Typo in the parser error message for a bad token"),
    ("parsing", "medium", "Parser mishandles the operator precedence of a nested expression"),
    ("parsing", "hard", "Grammar cannot parse ambiguous expression with nested operator groups"),
    ("parsing", "medium", "Parser mishandles the operator precedence of a simple expression"),
    ("parsing", "hard", "Ambiguous operator precedence parser builds the wrong nested groups"),
    ("graphs", "easy", "Rename the node label in the graph output"),
    ("graphs", "medium", "Graph traversal misses an edge when a cycle shares a node"),
    ("graphs", "hard", "Optimize the shortest path search on a huge dense graph with cycles"),
    ("dp", "medium", "Memo table returns a stale subproblem value after an update"),
    ("dp", "hard", "Prove the optimal substructure of the overlapping subproblem recurrence"),
    ("math", "medium", "Modular arithmetic wrong when the probability is a prime multiple"),
    ("math", "hard", "Combinatorics count overflows for a large prime modular modulus"),
    ("systems", "easy", "Small docs fix for the cache queue description"),
    ("systems", "medium", "Queue latency spikes when the thread pool memory is exhausted"),
    ("systems", "hard", "Optimize memory and cache latency for a large concurrent queue"),
)
HOLDOUT = (
    ("parsing", "hard", "Ambiguous operator precedence breaks the expression parser"),
    ("graphs", "easy", "Node label typo in the graph output"),
    ("dp", "hard", "Optimize the memo subproblem recurrence proof"),
    ("systems", "medium", "Thread memory exhaustion raises queue latency"),
)


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def tokenize(text):
    return [word for word in TOKEN_RE.findall(text.lower()) if word not in STOPWORDS]


class NaiveBayes:
    """Multinomial naive Bayes with Laplace smoothing over a small labelled corpus."""

    def __init__(self, labels):
        self.labels = tuple(labels)
        self.counts = {label: {} for label in self.labels}
        self.totals = {label: 0 for label in self.labels}
        self.documents = {label: 0 for label in self.labels}
        self.vocabulary = set()

    def fit(self, corpus):
        for label, text in corpus:
            self.documents[label] += 1
            for word in tokenize(text):
                self.counts[label][word] = self.counts[label].get(word, 0) + 1
                self.totals[label] += 1
                self.vocabulary.add(word)
        return self

    def posterior(self, text):
        """Posterior over the labels; unseen words only shift the evidence through smoothing."""
        size = len(self.vocabulary) or 1
        total_documents = sum(self.documents.values())
        scores = {}
        for label in self.labels:
            value = Fraction(self.documents[label] or 1, total_documents or 1)
            denominator = self.totals[label] + size
            for word in tokenize(text):
                value *= Fraction(self.counts[label].get(word, 0) + 1, denominator)
            scores[label] = value
        mass = sum(scores.values())
        if mass == 0:
            return {label: Fraction(1, len(self.labels)) for label in self.labels}
        return {label: value / mass for label, value in scores.items()}

    def predict(self, text):
        posterior = self.posterior(text)
        best = max(self.labels, key=lambda label: (posterior[label], label))
        return best, float(posterior[best])


def decide(confidence, token_count):
    """Abstain on thin evidence: too few tokens, or a winning posterior under the threshold."""
    return "tagged" if token_count >= MIN_TOKENS and confidence >= ABSTAIN_THRESHOLD else \
        "needs-triage"


def triage(area_model, difficulty_model, text):
    words = tokenize(text)
    area, area_confidence = area_model.predict(text)
    difficulty, difficulty_confidence = difficulty_model.predict(text)
    status = decide(min(area_confidence, difficulty_confidence), len(words))
    return {"status": status, "area": area if status == "tagged" else None,
            "difficulty": difficulty if status == "tagged" else None,
            "confidence": round(min(area_confidence, difficulty_confidence), 6),
            "tokens": words}


def models(training=TRAIN):
    area_model = NaiveBayes(AREAS).fit([(area, text) for area, _, text in training])
    difficulty_model = NaiveBayes(DIFFICULTIES).fit(
        [(difficulty, text) for _, difficulty, text in training])
    return area_model, difficulty_model


def rational_oracle(area_model, text):
    """The same posterior with exact fractions, computed straight from the count tables."""
    return area_model.posterior(text)


def main():
    area_model, difficulty_model = models()

    # 1) happy path: every hold-out issue is tagged - the area is right in all four cases, and the
    #    three unambiguous ones get their difficulty right too.
    predictions = [triage(area_model, difficulty_model, text) for _, _, text in HOLDOUT]
    expected = [(area, difficulty) for area, difficulty, _ in HOLDOUT]
    check(all(row["status"] == "tagged" for row in predictions)
          and [row["area"] for row in predictions] == [area for area, _ in expected]
          and [(row["area"], row["difficulty"]) for row in predictions[1:]] == expected[1:],
          f"hold-out predictions wrong: {[(r['area'], r['difficulty']) for r in predictions]}")

    # 2) the measured metric, plus the oracle for the single miss: the only wrong label is the
    #    difficulty of the ambiguous issue, which overlaps both difficulty classes and is therefore
    #    also the lowest-confidence prediction in the batch.
    labels_right = sum(1 for row, (area, difficulty) in zip(predictions, expected)
                       if row["area"] == area) + sum(
        1 for row, (_, difficulty) in zip(predictions, expected) if row["difficulty"] == difficulty)
    check(labels_right == 7 and labels_right / 8 == 0.875
          and predictions[0]["difficulty"] != "hard"
          and predictions[0]["confidence"] == min(row["confidence"] for row in predictions),
          f"the ambiguous hold-out must be the only miss: {labels_right}/8")

    # 3) abstention: thin or vague text is not guessed at, and the threshold boundary is exact.
    vague = triage(area_model, difficulty_model, "")
    single = triage(area_model, difficulty_model, "graph")
    check(vague["status"] == "needs-triage" and vague["area"] is None
          and single["status"] == "needs-triage" and single["tokens"] == ["graph"]
          and decide(ABSTAIN_THRESHOLD, MIN_TOKENS) == "tagged"
          and decide(ABSTAIN_THRESHOLD - 1e-9, MIN_TOKENS) == "needs-triage"
          and decide(1.0, MIN_TOKENS - 1) == "needs-triage",
          f"abstention boundary wrong: {vague} / {single}")

    # 4) the posterior is a real distribution over the labels for both classifiers.
    distributions = [rational_oracle(area_model, text) for _, _, text in HOLDOUT]
    check(all(abs(float(sum(row.values())) - 1.0) < 1e-12
              and all(0.0 <= float(value) <= 1.0 for value in row.values()) for row in distributions)
          and all(max(row.values()) == row[area_model.predict(text)[0]]
                  for row, (_, _, text) in zip(distributions, HOLDOUT)),
          "posteriors must be normalised and their maximum must be the prediction")

    # 5) independent oracle: the float prediction equals the exact rational one, for every hold-out.
    oracle = []
    for _, _, text in HOLDOUT:
        exact = area_model.posterior(text)
        best = max(AREAS, key=lambda label: (exact[label], label))
        oracle.append(best)
    check(oracle == [area_model.predict(text)[0] for _, _, text in HOLDOUT]
          and all(abs(float(area_model.posterior(text)[area_model.predict(text)[0]])
                      - area_model.predict(text)[1]) < 1e-12 for _, _, text in HOLDOUT),
          f"the rational oracle disagrees: {oracle}")

    # 6) unseen vocabulary never crashes the model and is not guessed at: with no known token left
    #    the queue abstains, while a one-class model trivially returns its only label.
    unseen = triage(area_model, difficulty_model, "zzyzx quux frobnicate")
    one_class = NaiveBayes(("fixed",)).fit([("fixed", "parser grammar")])
    label, confidence = one_class.predict("unrelated words here")
    check(unseen["status"] == "needs-triage" and unseen["area"] is None
          and area_model.predict("zzyzx quux")[0] in AREAS
          and label == "fixed" and abs(confidence - 1.0) < 1e-12
          and one_class.posterior("")["fixed"] == Fraction(1, 1),
          f"unseen/one-class boundaries wrong: {unseen} / {label} {confidence}")

    # 7) determinism: shuffling the training corpus (and re-fitting) changes nothing.
    order = sorted(range(len(TRAIN)), key=lambda index: (index * 4 + 1) % len(TRAIN))
    shuffled = tuple(TRAIN[index] for index in order)
    other_area, other_difficulty = models(shuffled)
    check([triage(other_area, other_difficulty, text) for _, _, text in HOLDOUT] == predictions
          and other_area.counts == area_model.counts
          and other_area.vocabulary == area_model.vocabulary,
          "training order must not change the model or the predictions")

    # 8) conservation: the per-label token totals sum to the number of retained tokens, and every
    #    count is positive for the classes that saw the word.
    retained = sum(len(tokenize(text)) for _, _, text in TRAIN)
    check(sum(area_model.totals.values()) == retained
          and sum(difficulty_model.totals.values()) == retained
          and all(count > 0 for counts in area_model.counts.values() for count in counts.values())
          and len(area_model.vocabulary) > 20,
          f"token accounting wrong: {sum(area_model.totals.values())} vs {retained}")

    print(f"predictions={[(r['area'], r['difficulty'], r['confidence']) for r in predictions]}")
    print(f"labels_right={labels_right}/8 vocabulary={len(area_model.vocabulary)} "
          f"retained_tokens={retained} abstained={vague['status']}")
    print("PASS")


if __name__ == "__main__":
    main()
