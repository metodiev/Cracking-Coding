def reversed_string_reversed(s):
    return ''.join(reversed(s))

def reversed_string_without_join(s):
    #this is not human readable it is going to print the address
    #something like <reversed object at 0x10464fb80>
    return reversed(s)

def reverse_string_without_join_using_loop(s):
    for char in reversed(s):
        print(char, end="")

def reverse_string_using_list(s):
    return list(reversed(s))

text = "I want to reverse this string 123. 321"
#print(reversed_string_reversed(text))
#print(reversed_string_without_join(text))
#print(reverse_string_without_join_using_loop(text))
print(reverse_string_using_list(text))