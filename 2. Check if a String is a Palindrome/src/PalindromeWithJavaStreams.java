import java.util.stream.IntStream;

public class PalindromeWithJavaStreams {

    public static boolean isPalindrome(String str){
        return IntStream.range(0, str.length() / 2)
                .allMatch( i -> str.toLowerCase().charAt(i) == str.toLowerCase().charAt(str.length() -i - 1));
    }

    public static void main(String[] args) {
        // palindrome function with java streams
        String str = "Detartrated";

        boolean isPalindrome = isPalindrome(str);
        System.out.println("Is palindrome:"+ isPalindrome);
    }
}
