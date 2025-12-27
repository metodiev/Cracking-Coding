def is_palindrome_reversed(s):
    s = s.lower().replace(" ", "")
    return s == ''.join(reversed(s))

text = "Radar"
print(is_palindrome_reversed(text))