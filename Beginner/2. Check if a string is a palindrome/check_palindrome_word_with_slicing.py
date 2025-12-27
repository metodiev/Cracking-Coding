def is_palindrome_slacing(s):
    s = s.lower().replace(" ", "") #ignore whitespaces and capital letters
    return s == s[::-1]

text = "Radar"
print(is_palindrome_slacing(text))
