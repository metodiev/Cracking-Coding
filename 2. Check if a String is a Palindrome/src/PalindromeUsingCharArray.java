public class PalindromeUsingCharArray {

    public static boolean isPalindrome(String str){
        char [] chars = str.toLowerCase().toCharArray();
        int i = 0;
        int j = chars.length -1;

        while (i < j){
            if (chars[i] != chars[j]) {
                return false;
            }
        }
        return true;

    }

    public static void main(String[] args) {
        //call ispalindrome using to char array
        String str = "Detartrated";
        boolean isPalindrome = isPalindrome(str);
        System.out.println("is palindrome");
        System.out.println("Is palindrome " + isPalindrome);
    }
}
