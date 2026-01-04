import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ReverseAStringUsingCollectionReverse {
    public static String reverseAString(String str){
        List<Character> chars = new ArrayList<>();
        for (char c : str.toCharArray()) {
            chars.add(c);
        }
        Collections.reverse(chars);

        StringBuilder reversed = new StringBuilder();
        for (char c : chars){
            reversed.append(c);
        }

        return reversed.toString();
    }

    public static void main(String[] args) {
        //call method to reverse the string
        String str = "String to be reversed, W 123 31 2";
        String reversedString = reverseAString(str);

        System.out.println(reversedString);
    }
}
