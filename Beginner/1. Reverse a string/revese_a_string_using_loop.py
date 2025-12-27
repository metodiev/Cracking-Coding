def reverse_string_in_loop(s):
    reversed_s = ""
    for char in s:
        reversed_s = char + reversed_s

    return reversed_s

text = "I want to reverst this string. 123 string 312" 
print(reverse_string_in_loop(text))
