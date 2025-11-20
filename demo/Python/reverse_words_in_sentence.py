def reverse_words(s: str) -> str:
    return " ".join(s.split()[::-1])

print(reverse_words("the sky is blue the river is red"))