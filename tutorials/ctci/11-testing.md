# Chapter 11 · Testing

> Book reference: **Ch 11 — Testing** (testing a real-world object, a piece of software, a function; troubleshooting).

Test questions come in two flavours: *“test this thing”* (design a test plan) and *“your test failed —
debug it”*. Both reward methodical thinking, not trivia.

```mermaid
flowchart TD
    A["What are you testing?"] --> B{"Real-world object?"}
    B -->|yes| C["Who uses it, for what?<br/>Enumerate flows & failure modes"]
    B -->|no| D{"Piece of software?"}
    D -->|yes| E["Inputs → outputs, state,<br/>integration points, perf"]
    D -->|no| F["A function: normal / edge / error cases"]
```

---

## 1. The test pyramid

```mermaid
flowchart TD
    E2E["End-to-end (few) — whole flows<br/>slow, brittle, high confidence"]
    INT["Integration (some) — modules together<br/>DB, queues, HTTP"]
    UNIT["Unit (many) — one function/class<br/>fast, deterministic"]
    UNIT --> INT --> E2E
```

| Level | Speed | Confidence | When it fails you know… |
|-------|-------|------------|-------------------------|
| unit | ms | narrow | exactly where |
| integration | 100ms–s | medium | which boundary broke |
| e2e | s–min | broad | something broke, somewhere |

---

## 2. Anatomy of a test — Arrange / Act / Assert

```mermaid
sequenceDiagram
    participant T as Test
    participant S as System under test
    T->>T: Arrange — build inputs & state
    T->>S: Act — call the operation once
    S-->>T: result
    T->>T: Assert — compare with expected
```

```java
// Java — plain assertions, no framework needed for practice
void testReverse() {
    assertEquals("olleh", Solution.reverse("hello"));
    assertEquals("", Solution.reverse(""));              // edge: empty
    assertEquals("a", Solution.reverse("a"));            // edge: single char
    assertEquals("aa", Solution.reverse("aa"));          // edge: duplicates
}
```

```python
# Python — pytest style
import pytest
from solution import reverse

@pytest.mark.parametrize("raw, expected", [
    ("hello", "olleh"),
    ("", ""),            # empty
    ("a", "a"),          # single
    ("ab", "ba"),        # even
])
def test_reverse(raw, expected):
    assert reverse(raw) == expected
```

```csharp
// C# — xUnit
[Theory]
[InlineData("hello", "olleh")]
[InlineData("", "")]
public void Reverse_Works(string raw, string expected)
    => Assert.Equal(expected, Solution.Reverse(raw));
```

```javascript
// JavaScript — node:assert
const assert = require("node:assert");
const { reverse } = require("./solution");

assert.strictEqual(reverse("hello"), "olleh");
assert.strictEqual(reverse(""), "");
console.log("PASS");
```

---

## 3. A function test plan — the four buckets

```mermaid
mindmap
  root((Test buckets))
    Normal
      typical inputs
      multiple sizes
    Edge
      empty / null
      single element
      min / max values
      duplicates
    Error
      invalid types
      out of range
      malformed input
    Performance
      n = 10^5 / 10^6
      worst-case shape
      memory growth
```

For every task in this repo, the four-bucket scan is exactly what the self-check blocks implement.

---

## 4. Troubleshooting playbook

```mermaid
flowchart TD
    F["Failure / bug report"] --> R["Reproduce — same input, same seed"]
    R --> N["Narrow: bisect commits, inputs, layers"]
    N --> I["Isolate: minimal failing case"]
    I --> H["Hypothesize → instrument → verify"]
    H --> P["Fix + regression test that fails before, passes after"]
    P --> S["Sanity: run the full suite for unintended breakage"]
```

| Symptom | First suspects |
|---------|----------------|
| passes locally, fails in CI | timezone, locale, seed, parallelism, unordered maps |
| flaky occasionally | shared mutable state, timing, random without seed |
| correct for small n only | integer overflow, recursion depth, cache limits |
| silently wrong | floating point equality, off-by-one on boundaries |

---

## 5. Fuzzing & property-based testing

```mermaid
flowchart LR
    G["Generate random input"] --> R["Run system"]
    R --> C{"Property holds?<br/>e.g. sort output is ordered,<br/>reverse(reverse(x)) == x"}
    C -->|no| F["Shrink input → minimal counterexample"]
    C -->|yes| G
```

Great properties to check on this repo's own solutions:

- `reverse(reverse(s)) == s`
- sorting output is non-decreasing and a permutation of the input
- Dijkstra distance ≤ Bellman-Ford distance on the same non-negative graph
- serialize → parse round-trips to an equal object

Mapped tasks: [750 fuzzing harness](../../tasks/expert/750-implement-fuzzing-harness-with-coverage-guided-mutation-afl-like.md),
[751 grammar fuzzing](../../tasks/expert/751-implement-grammar-based-fuzzing-for-structured-inputs.md),
[957 randomized test-case generator](../../tasks/master/957-implement-randomized-test-case-generator-with-oracle-checker-for.md),
[952 autograder](../../tasks/master/952-implement-autograder-that-runs-tests-measures-complexity-and-rep.md),
[958 plagiarism detector](../../tasks/master/958-implement-plagiarism-detector-comparing-code-structure-and-logic.md),
[940 load generator](../../tasks/master/940-implement-high-fidelity-load-generator-with-correlated-user-beha.md),
[891 failure-injection benchmarking](../../tasks/master/891-implement-benchmarking-harness-for-large-distributed-systems-wit.md).

> Full list: [ctci-mapping.md → Ch 11](../ctci-mapping.md).

## Interview checklist for “test this”

```mermaid
flowchart LR
    A["Clarify the spec & users"] --> B["List inputs, outputs, states"]
    B --> C["Enumerate normal / edge / error / perf"]
    C --> D["Prioritize by risk"]
    D --> E["Describe automation level for each"]
```
