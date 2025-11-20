def recursive_reverse(s: str) -> str:
    if len(s) == 0:
        return s
    return recursive_reverse(s[1:]) + s[0]

print(recursive_reverse("hello"))
