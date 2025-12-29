# Reverse a String in JavaScript

This document lists **multiple ways to reverse a string** in JavaScript. Each method demonstrates a different approach, from basic to advanced.

---

## 1. Using Built-in `split()`, `reverse()`, and `join()`

```javascript
function reverseString1(str) {
  return str.split('').reverse().join('');
}

// Example
console.log(reverseString1("hello")); // Output: "olleh"
```
## 2. Using a for loop (iterative approach)

```javascript
function reverseString2(str) {
  let reversed = '';
  for (let i = str.length - 1; i >= 0; i--) {
    reversed += str[i];
  }
  return reversed;
}

// Example
console.log(reverseString2("world")); // Output: "dlrow"

```

Explanation:
Iterates the string from the end to the beginning, appending each character to a new string.

## 3. Using for...of loop and array unshift()

```javascript
function reverseString3(str) {
  const reversedArray = [];
  for (const char of str) {
    reversedArray.unshift(char);
  }
  return reversedArray.join('');
}

// Example
console.log(reverseString3("JavaScript")); // Output: "tpircSavaJ"

```

Explanation:
unshift() adds each character to the front of a new array, effectively reversing it.

## 4. Using Recursion

```javascript
function reverseString4(str) {
  if (str === "") return "";
  return reverseString4(str.substr(1)) + str[0];
}

// Example
console.log(reverseString4("recursion")); // Output: "noisrucer"

```

Explanation:

Calls the function on the substring (excluding the first character)
Appends the first character at the end on each recursive step

## 5. Using reduce() on an array

```javascript
function reverseString5(str) {
  return str.split('').reduce((rev, char) => char + rev, '');
}

// Example
console.log(reverseString5("reduce")); // Output: "ecuder"

```

Explanation:

reduce() accumulates a reversed string by prepending each character.

## 6. Using ES6 spread operator

```javascript
const reverseString6 = str => [...str].reverse().join('');

console.log(reverseString6("ES6")); // Output: "6SE"

```

Explanation:

[...str] spreads the string into an array

Then reverse and join, similar to split() method

## Summary  

| Method                 | Type       | Notes                                        |
| ---------------------- | ---------- | -------------------------------------------- |
| split() + reverse() + join() | Built-in   | Quickest and easiest                         |
| for loop               | Iterative  | No extra arrays                              |
| for...of + unshift()   | Iterative  | Uses array unshift                           |
| Recursion              | Recursive  | Elegant, may hit stack limit on long strings |
| reduce()               | Functional | Functional programming style                 |
| Spread + reverse       | ES6        | Modern syntax, clean                         |
