import java.util.ArrayDeque;
import java.util.Deque;

public class PalindromeUsingDeque {
    public static boolean isPalindrome(String str){
        Deque<Character> deque = new ArrayDeque<>();
        for (char c : str.toCharArray()){
            deque.add(c);
        }

        while (deque.size() > 1 ){
            if (!deque.pollFirst().equals(deque.pollLast())){
                return false;
            }
        }
        return true;
    }
}
