import java.util.*;
import java.util.Arrays.*;

public class ReverseWordsInSentence {

    public static String reverseWords(String s){
        String[] words = s.trim().split("\\s+");
        Collections.reverse(Arrays.asList(words));
        return String.join(" ", words);

    }

    public static void main(String[] args){

        System.out.println(reverseWords("The sky is blue, the river is red"));
    }
}