# Two-Sum Unique Pairs in JavaScript

This project demonstrates **different ways to find all unique pairs of numbers in an array** that sum up to a target value using JavaScript.


## Task Description

**Problem:**  
Given an array of integers `nums` and a target integer `target`, return **all unique pairs** `[a, b]` such that:

1. `a + b === target`
2. Each pair appears **only once**, regardless of order (`[a, b]` is the same as `[b, a]`)

**Example:**

```javascript
Input: nums = [1, 2, 3, 2, 4, 5], target = 5
Output: [[1, 4], [2, 3]]
```

## 1. Using a Set for seen numbers

```javascript
function twoSumUniquePairs1(nums, target) {
  const seen = new Set();
  const pairs = new Set();

  for (let num of nums) {
    const complement = target - num;
    if (seen.has(complement)) {
      // Sort pair to avoid duplicates
      const sortedPair = [num, complement].sort((a,b) => a-b);
      pairs.add(sortedPair.toString());
    }
    seen.add(num);
  }

  // Convert set of strings back to array of pairs
  return Array.from(pairs).map(pair => pair.split(',').map(Number));
}

// Example
console.log(twoSumUniquePairs1([1,2,3,2,4,5], 5)); // Output: [[1,4],[2,3]]

```

## 2. Using Nested Loops (Brute Force)

```javascript
function twoSumUniquePairs2(nums, target) {
  const pairs = [];
  for (let i = 0; i < nums.length; i++) {
    for (let j = i + 1; j < nums.length; j++) {
      if (nums[i] + nums[j] === target) {
        const pair = [nums[i], nums[j]].sort((a,b) => a-b);
        // Check if pair is already in pairs
        if (!pairs.some(p => p[0] === pair[0] && p[1] === pair[1])) {
          pairs.push(pair);
        }
      }
    }
  }
  return pairs;
}

console.log(twoSumUniquePairs2([1,2,3,2,4,5], 5)); // Output: [[1,4],[2,3]]

```

## 3. Using Map for Counting Elements

```javascript
function twoSumUniquePairs3(nums, target) {
  const count = new Map();
  const result = [];

  for (let num of nums) {
    count.set(num, (count.get(num) || 0) + 1);
  }

  for (let num of count.keys()) {
    const complement = target - num;
    if (count.has(complement)) {
      if (num === complement && count.get(num) > 1 || num !== complement) {
        result.push([num, complement].sort((a,b) => a-b));
      }
    }
  }

  // Remove duplicates
  const unique = Array.from(new Set(result.map(p => p.toString())))
                      .map(s => s.split(',').map(Number));
  return unique;
}

console.log(twoSumUniquePairs3([1,2,3,2,4,5], 5)); // Output: [[1,4],[2,3]]

```