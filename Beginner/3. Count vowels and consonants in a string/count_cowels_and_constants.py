def count_vowels_and_constns(s):
    vowels = set("aeiou")
    vowel_count = 0 
    consonant_count = 0

    for ch in s.lower():
        if ch.isalpha(): # we are checking only letters
            if ch in vowels:
                vowel_count += 1
            else:
                consonant_count += 1

    return vowel_count, consonant_count

text = "Hello World"
vowels, consonants = count_vowels_and_constns(text)
print("Vowels:", vowels)
print("Consonants:", consonants)
