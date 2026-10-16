"""Task 975 - Implement natural-language problem paraphrasing generator for diverse practice.  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)

A problem statement is a sentence skeleton with typed slots (framing, action, answer object, relation,
closing remark, plus the numeric parameters of one instance).  Every slot has a set of interchangeable
alternatives that all carry the same meaning class, so the generator can draw new phrasings from a seeded
PRNG while the problem keeps its meaning.  Meaning is checked the other way round from generation: a
lexicon scanner re-extracts the class of every phrase together with the numbers from a rendered variant
and the *signature* must equal the source's, so a swapped relation, a different action or an altered
parameter is detected instead of shipped.

Simplified versus an LLM paraphraser: a skeleton plus hand-written slot alternatives instead of free-form
generation, so the candidate space is finite (432 statements here) and can be enumerated.  Verified: the
signature round-trip on every variant, membership of the generated variants in the independently
enumerated candidate space, distinctness and style coverage, a Jaccard diversity floor, planted
meaning-breaking edits being rejected, empty/singleton spaces, and seeded reproducibility.
"""
import re

SKELETON = ("{intro} an array of {count} distinct integers, {action} the {object} of the two entries "
            "that {relation} {target}. {closing}")
SLOTS = {
    "intro": ("Given", "You are given", "Consider", "Take"),
    "action": ("return", "find and return", "produce", "report"),
    "object": ("indices", "positions", "index pair"),
    "relation": ("add up to", "sum to", "have a total of"),
    "closing": ("Report them in any order.", "Any valid pair is accepted.",
                "The order of the answer does not matter."),
}
CANONICAL = {"intro": "Given", "action": "return", "object": "indices",
             "relation": "add up to", "closing": "Report them in any order."}
# phrase -> meaning class; the scanner must recover the same classes for every valid paraphrase
LEXICON = {
    "intro": ("Given", "You are given", "Consider", "Take"),
    "action": ("return", "find and return", "produce", "report", "count"),
    "object": ("indices", "positions", "index pair"),
    "relation": ("add up to", "sum to", "have a total of", "multiply to"),
    "closing": ("Report them in any order.", "Any valid pair is accepted.",
                "The order of the answer does not matter."),
}
CLASSES = {
    "intro": {"Given": "supplied_input", "You are given": "supplied_input", "Consider": "supplied_input",
              "Take": "supplied_input"},
    "action": {"return": "emit", "find and return": "emit", "produce": "emit", "report": "emit",
               "count": "tally"},
    "object": {"indices": "answer_position", "positions": "answer_position", "index pair": "answer_position"},
    "relation": {"add up to": "sum", "sum to": "sum", "have a total of": "sum", "multiply to": "product"},
    "closing": {"Report them in any order.": "free_order", "Any valid pair is accepted.": "free_order",
                "The order of the answer does not matter.": "free_order"},
}
PARAMETERS = {"count": 8, "target": 13}
MASK64 = (1 << 64) - 1


def check(ok, message):
    if not ok:
        raise AssertionError(message)


class Rng:
    """splitmix64 - fixed seed, so the drawn paraphrases are reproducible."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def below(self, bound):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        return (value ^ (value >> 31)) % bound


def pick(rng, options):
    return options[rng.below(len(options))]


def render(choices):
    return SKELETON.format(**choices)


def canonical_choices(number, target):
    return dict(CANONICAL, count=str(number), target=str(target))


def source_statement(number=PARAMETERS["count"], target=PARAMETERS["target"]):
    return render(canonical_choices(number, target))


def candidate_space(number=PARAMETERS["count"], target=PARAMETERS["target"]):
    """Independent oracle material: every statement the skeleton can produce for these parameters."""
    space = set()
    for intro in SLOTS["intro"]:
        for action in SLOTS["action"]:
            for answer in SLOTS["object"]:
                for relation in SLOTS["relation"]:
                    for closing in SLOTS["closing"]:
                        space.add(render({"intro": intro, "action": action, "object": answer,
                                          "relation": relation, "closing": closing,
                                          "count": str(number), "target": str(target)}))
    return space


def signature(text):
    """Meaning recovered from a rendered statement: one class per slot family, plus the parameters."""
    lowered = text.lower()
    classes = tuple(tuple(sorted({CLASSES[family][phrase] for phrase in LEXICON[family]
                                  if phrase.lower() in lowered})) for family in
                    ("intro", "action", "object", "relation", "closing"))
    return classes + (tuple(int(token) for token in re.findall(r"\d+", text)),)


def validate(text, number=PARAMETERS["count"], target=PARAMETERS["target"]):
    """Reason strings for every meaning drift between `text` and the source instance (empty == fine)."""
    expected = signature(source_statement(number, target))
    found = signature(text)
    reasons = []
    for position, family in enumerate(("intro", "action", "object", "relation", "closing")):
        if found[position] != expected[position]:
            reasons.append(f"{family}: {found[position]} != {expected[position]}")
    if found[-1] != expected[-1]:
        reasons.append(f"parameters: {found[-1]} != {expected[-1]}")
    return reasons


def paraphrase(number=PARAMETERS["count"], target=PARAMETERS["target"], wanted=6, seed=7):
    """Draw `wanted` distinct paraphrases of one instance, never repeating the canonical phrasing."""
    rng = Rng(seed)
    source = source_statement(number, target)
    variants, seen = [], {source}
    for _ in range(200):
        if len(variants) >= wanted:
            break
        choices = {"count": str(number), "target": str(target)}
        choices.update({family: pick(rng, SLOTS[family]) for family in SLOTS})
        text = render(choices)
        if text not in seen:
            seen.add(text)
            variants.append(text)
    return variants


def tokens(text):
    return set(re.findall(r"[a-z]+", text.lower()))


def distance(left, right):
    """Jaccard distance between the word sets of two statements - how different they read."""
    left_tokens, right_tokens = tokens(left), tokens(right)
    union = left_tokens | right_tokens
    return 1.0 - len(left_tokens & right_tokens) / len(union) if union else 0.0


def mean_distance(texts):
    pairs = [(left, right) for index, left in enumerate(texts) for right in texts[index + 1:]]
    return sum(distance(left, right) for left, right in pairs) / len(pairs) if pairs else 0.0


def styles(variants):
    """Which framing/action phrases the draw actually used - the style coverage of the batch."""
    return {"intro": {phrase for phrase in SLOTS["intro"] for text in variants if text.startswith(phrase)},
            "action": {phrase for phrase in SLOTS["action"] for text in variants
                       if f", {phrase} the" in text}}


def main():
    source = source_statement()
    variants = paraphrase(wanted=6, seed=7)
    # 1) the scanner recovers the source meaning: one class per family plus both parameters.
    check(signature(source) == (("supplied_input",), ("emit",), ("answer_position",), ("sum",),
                               ("free_order",), (8, 13))
          and len(validate(source)) == 0 and re.findall(r"\d+", source) == ["8", "13"],
          f"the source signature is wrong: {signature(source)}")
    # 2) generation: six distinct variants, none of them the canonical phrasing, all validated.
    check(len(variants) == 6 and len(set(variants)) == 6 and source not in variants
          and all(not validate(text) for text in variants),
          f"variant generation is wrong: {len(set(variants))} distinct of {len(variants)}")
    # 3) meaning round-trip: every variant recovers exactly the source signature.
    check(all(signature(text) == signature(source) for text in variants)
          and len({signature(text) for text in variants}) == 1,
          f"a variant changed the meaning: {[text for text in variants if signature(text) != signature(source)]}")
    # 4) independent oracle: every variant is a member of the enumerated candidate space.
    space = candidate_space()
    check(len(space) == 4 * 4 * 3 * 3 * 3 == 432 and set(variants) <= space and source in space
          and all(text.count("{") == 0 for text in space),
          f"a variant is outside the enumerated candidate space: {sorted(set(variants) - space)}")
    # 5) diversity: the batch reads differently and covers several framings and actions.
    coverage = styles(variants)
    check(mean_distance(variants) >= 0.1 and len(coverage["intro"]) >= 3 and len(coverage["action"]) >= 3
          and min(distance(left, right) for left in variants for right in variants if left != right) > 0.02,
          f"the batch is not diverse enough: {coverage} distance={mean_distance(variants):.3f}")
    # 6) planted meaning breaks are rejected: swapped relation, different action, altered parameter,
    #    dropped closing remark - while a harmless rewording of the same meaning is accepted.
    broken = [render(dict(canonical_choices(8, 13), relation="multiply to")),
              render(dict(canonical_choices(8, 13), action="count")),
              source.replace(" 13.", " 14."),
              source.replace(" Report them in any order.", ""),
              source.replace("the two entries", "those two entries")]
    reasons = [validate(text) for text in broken]
    check(all(reasons[:4]) and "relation" in reasons[0][0] and "action" in reasons[1][0]
          and "parameters" in reasons[2][0] and "closing" in reasons[3][0] and reasons[4] == [],
          f"meaning drift was not detected: {reasons}")
    # 7) boundary: an empty request yields nothing, an oversized request is capped by the candidate
    #    space and the attempt budget rather than looping forever, and blank text never validates.
    oversized = paraphrase(wanted=999, seed=3)
    check(paraphrase(wanted=0) == [] and 0 < len(oversized) < 999 and len(set(oversized)) == len(oversized)
          and all(not validate(text) for text in oversized) and set(oversized) <= space
          and validate("") != [] and signature("")[-1] == () and distance("", "") == 0.0
          and mean_distance([]) == 0.0,
          f"the empty and oversized requests are wrong: {len(oversized)} variants")
    # 8) reproducibility: the same seed repeats the batch exactly, another seed still preserves meaning.
    other = paraphrase(wanted=6, seed=11)
    check(paraphrase(wanted=6, seed=7) == variants and other != variants and len(set(other)) == 6
          and all(not validate(text) for text in other) and other and set(other) <= space,
          "paraphrase generation must be reproducible and seed-sensitive")
    # 9) validator accounting: the whole batch plus the planted breaks is classified consistently.
    accepted = [text for text in variants + other if not validate(text)]
    rejected = [text for text in broken if validate(text)]
    check(len(accepted) == 12 and len(rejected) == 4 and len(accepted) + len(broken) == 17
          and all(not validate(text) for text in accepted)
          and sorted(signature(text) for text in accepted) == sorted([signature(source)] * 12),
          f"validator accounting is wrong: accepted={len(accepted)} rejected={len(rejected)}")
    print(f"variants={len(variants)} space={len(space)} signatures={len({signature(t) for t in variants})} "
          f"mean_distance={mean_distance(variants):.3f} intros={len(coverage['intro'])}")
    print(f"rejected_planted={len(rejected)}/{len(broken)} source={source!r}")
    print("PASS")


if __name__ == "__main__":
    main()
