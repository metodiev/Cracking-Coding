
"""Task 127 - Decode ways (num decodings) DP. (Tier: Beginner, CTCI Ch 8)

'A'..'Z' map to "1".."26". dp[i] = ways to decode s[:i].
dp[i] += dp[i-1] if s[i-1] is a valid single digit (1-9);
dp[i] += dp[i-2] if the two-digit number s[i-2:i] is 10..26.
Leading zeros and zeros inside pairs make strings invalid.
"""


def num_decodings(s):
    if not s:
        return 0
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 0 if s[0] == "0" else 1
    for i in range(2, n + 1):
        one = int(s[i - 1])
        two = int(s[i - 2:i])
        if 1 <= one <= 9:
            dp[i] += dp[i - 1]
        if 10 <= two <= 26:
            dp[i] += dp[i - 2]
    return dp[n]


def main():
    assert num_decodings("12") == 2          # AB, L
    assert num_decodings("226") == 3         # BZ, VF, BBF
    assert num_decodings("06") == 0          # leading zero
    assert num_decodings("10") == 1          # J only
    assert num_decodings("27") == 1          # BG (27 invalid pair)
    assert num_decodings("0") == 0
    assert num_decodings("11106") == 2       # AAJF, K1F? no: 1-1-10-6, 11-10-6
    assert num_decodings("") == 0


if __name__ == "__main__":
    main()
    print("PASS")
