# Check if a String is a Palindrome in Python

A **palindrome** is a string that reads the same forwards and backwards, ignoring spaces, punctuation, and sometimes capitalization.  
This project demonstrates how to check if a string is a palindrome using Python.

---

## Example

```python
text = "radar"
# This is a palindrome
```
Methods to Check for a Palindrome

## 1. Using Slicing

Python slicing can be used to reverse the string and compare it to the original.

```python
def is_palindrome_slicing(s):
    s = s.lower().replace(" ", "")  # optional: ignore case and spaces
    return s == s[::-1]

text = "Radar"
print(is_palindrome_slicing(text))  # Output: True
```

Explanation:

s[::-1] reverses the string.

Compare the reversed string to the original (normalized) string.

## 2. Using a Loop

You can check each character from the start and end step by step.

```python 
def is_palindrome_loop(s):
    s = s.lower().replace(" ", "")
    length = len(s)
    for i in range(length // 2):
        if s[i] != s[length - 1 - i]:
            return False
    return True

text = "Radar"
print(is_palindrome_loop(text))  # Output: True
```
Explanation:

Compare the first and last characters, then second and second-last, etc.

If any pair doesn’t match → not a palindrome.

length // 2
// -> floor division

Result is an integer

Any decimal part is discarded

Examples:
```python

5 // 2  # 2
6 // 2  # 3
7 // 2  # 3
```
## 3. Using reversed() Function

You can also use reversed() and join() to create the reversed string.

```python
def is_palindrome_reversed(s):
    s = s.lower().replace(" ", "")
    return s == ''.join(reversed(s))

text = "Radar"
print(is_palindrome_reversed(text))  # Output: True
```

Explanation:

reversed(s) returns an iterator of characters in reverse order.

''.join(reversed(s)) converts it back into a string to compare.


## Summary

- A palindrome reads the same forwards and backwards.
- Slicing is the simplest Pythonic way.
- Loops and reversed() are more explicit and teach how comparison works internally.