# Suffix Array Implementation in JavaScript

This project demonstrates how to **construct a Suffix Array (SA)** for a string using **efficient algorithms** like **Doubling Method** or **SA-IS**. Suitable for **medium-sized input strings**.

---

## 📌 Task Description

**Problem:**
Given a string `s` of length `N` (medium size), construct a **Suffix Array (SA)** — an array of starting indices of all suffixes of `s` in **lexicographical order**.

**Example:**

```javascript
Input: s = "banana"
Output: [5, 3, 1, 0, 4, 2]
// Explanation:
// suffixes: ["a", "ana", "anana", "banana", "na", "nana"]
// sorted:   ["a","ana","anana","banana","na","nana"]
// indices:  [5, 3, 1, 0, 4, 2]
```

## 1. Doubling Method (Iterative Ranking)

```javascript
function buildSuffixArrayDoubling(s) {
  const n = s.length;
  let sa = Array.from({length: n}, (_, i) => i);
  let rank = Array.from(s).map(c => c.charCodeAt(0));
  let k = 1;

  while (k < n) {
    sa.sort((a, b) => {
      if (rank[a] !== rank[b]) return rank[a] - rank[b];
      const rankA = a + k < n ? rank[a + k] : -1;
      const rankB = b + k < n ? rank[b + k] : -1;
      return rankA - rankB;
    });

    const tmp = Array(n);
    tmp[sa[0]] = 0;
    for (let i = 1; i < n; i++) {
      tmp[sa[i]] = tmp[sa[i - 1]] +
        (rank[sa[i - 1]] !== rank[sa[i]] ||
         ((sa[i - 1] + k < n ? rank[sa[i - 1] + k] : -1) !== (sa[i] + k < n ? rank[sa[i] + k] : -1)) ? 1 : 0);
    }
    rank = tmp;
    k <<= 1; // multiply k by 2
  }

  return sa;
}

// Example
console.log(buildSuffixArrayDoubling("banana")); // Output: [5,3,1,0,4,2]

```

Explanation:

- Initially ranks each character.
- Sorts suffixes based on first k characters iteratively doubling k.
- Runs in O(N log² N) — suitable for medium N.

## 2. SA-IS Algorithm (Induced Sorting)

 Note: SA-IS is complex to implement from scratch. Below is a high-level idea:

Classify characters as S-type or L-type.
- Identify LMS substrings.
- Recursively sort LMS substrings.
- Induce the order of remaining suffixes.
- Practical JavaScript implementation for medium N often uses Doubling Method, while SA-IS is better for very large strings.

## Example Usage

```javascript
const s = "banana";
const sa = buildSuffixArrayDoubling(s);
console.log(sa); // Output: [5,3,1,0,4,2]

// Access sorted suffixes
const sortedSuffixes = sa.map(i => s.substring(i));
console.log(sortedSuffixes); // ["a","ana","anana","banana","na","nana"]

```