def reverse_char_array(s: list) -> None:
    left, right = 0, len(s) - 1
    while left < right:
        s[left], s[right] = s[right], s[left]
        left +=1
        right -=1
arr = ['h', 'e', 'l', 'l', 'o']        
reverse_char_array(arr)
print(arr)