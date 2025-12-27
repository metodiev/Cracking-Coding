# Reverse a String in Python

This project demonstrates how to **reverse a string** in Python. Reversing a string is a common programming task and can be done in multiple ways.

---

## Example

```python
original_string = "Hello, World!"
reversed_string = "!"dlroW ,olleH"
```

## Methods to Reverse a String
1. Using Slicing

Python strings support slicing, which can be used to reverse a string easily.

```python 
def reverse_string_slicing(s):
    return s[::-1]

text = "Python"
print(reverse_string_slicing(text))  # Output: nohtyP
```

Explanation:

s[start:stop:step] is slicing syntax.

[::-1] means start from the end towards the beginning, effectively reversing the string.

## 2. Using the reversed() Function

Python provides a built-in function reversed() that returns an iterator that can be converted to a string.

```python 
def reverse_string_reversed(s):
    return ''.join(reversed(s))

text = "Python"
print(reverse_string_reversed(text))  # Output: nohtyP
```

Explanation:

reversed(s) returns an iterator over the string in reverse order.

''.join() combines the characters back into a string.

## 3. Using a Loop

You can also reverse a string using a simple for loop.


```python
def reverse_string_loop(s):
    reversed_s = ""
    for char in s:
        reversed_s = char + reversed_s
    return reversed_s

text = "Python"
print(reverse_string_loop(text))  # Output: nohtyP
```

Explanation:

Start with an empty string.
Prepend each character to build the reversed string.

reversed_s = char + reversed_s

This is the key step.

Instead of adding the character at the end, we add it at the beginning.

This “pushes” all previous characters to the right, effectively reversing the order.

| Iteration | `char` | `reversed_s` before | `reversed_s` after |
| --------- | ------ | ------------------- | ------------------ |
| 1         | 'P'    | ""                  | "P"                |
| 2         | 'y'    | "P"                 | "yP"               |
| 3         | 't'    | "yP"                | "tyP"              |
| 4         | 'h'    | "tyP"               | "htyP"             |
| 5         | 'o'    | "htyP"              | "ohtyP"            |
| 6         | 'n'    | "ohtyP"             | "nohtyP"           |


## Summary

- Python makes reversing a string easy and flexible.
- Use slicing for a short and Pythonic approach.
- Use reversed() or a loop if you want more explicit control.