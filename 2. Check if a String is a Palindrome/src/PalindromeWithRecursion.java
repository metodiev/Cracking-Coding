public class PalindromeWithRecursion {

    public static boolean isPalindrome(String str){
        return isPalindrome(str.toLowerCase(), 0, str.length() -1);
    }

    private static boolean isPalindrome(String s, int left, int right ){
        if (left >= right){
            return true;

        }
        if (s.charAt(left) != s.charAt(right)){
            return false;
        }
        return isPalindrome(s, left + 1, right -1);
    }

    public static void main(String[] args) {
//call palindrome function with recursion
        String str = "Detartrated";
        boolean isPalindrome = isPalindrome(str);

        System.out.println("Detartrated is palindrome:" + isPalindrome);
    }
}
