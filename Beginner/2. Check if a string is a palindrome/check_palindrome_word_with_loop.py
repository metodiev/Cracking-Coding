def is_palindrome_check_with_loop(s):
    s = s.lower().replace(" ", "")
    length = len(s)
    for i in range(length // 2):
        if s[i] != s[length - 1 - i]:
            return False
    return True

text = "Radar"
print(is_palindrome_check_with_loop(text))


def test_floor_division():
    for i in range(101):
        print(f"We have for i:", i)
        print(f"The floor division is:", i //2)

#print(test_floor_division())
