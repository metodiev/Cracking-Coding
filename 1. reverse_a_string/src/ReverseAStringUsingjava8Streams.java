import java.util.stream.Collectors;

public class ReverseAStringUsingjava8Streams {
    public static String reverseAstring(String str){
        String  reversedString = new StringBuilder(
                str.chars()
                        .mapToObj(c -> (char)c)
                        .map(String::valueOf)
                        .collect(Collectors.joining())
        ).reverse().toString();

        return  reversedString;
    }

    public static void main(String[] args) {
        //call reversed String function
        String str = "String to be reversed, Hello, W 123 321";
        String reversedString = reverseAstring(str);
        System.out.println(reversedString);
    }
}
