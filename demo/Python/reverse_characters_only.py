def reverse_ignore_spaces(s: str) -> str:
    arr = list(s)
    left, right = 0, len(arr) -1 

    while left < right:
        if arr[left] == ' ':
            left +=1
        elif arr[right] == ' ':
            right-=1
        else:
            arr[left], arr[right], = arr[right], arr[left]
            left +=1
            right -=1

    return ''.join(arr)

if __name__ == "__main__":
    print(reverse_ignore_spaces("a b c"))

