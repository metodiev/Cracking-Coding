public class RecursiveReverse{

    public static String reverseRecursively(String s){
        if(s.isEmpty()) return s;
        return reverseRecursively(s.substring(1) + s.charAt(0));
    }

    public static void main(String[] args){
        System.out.println(reverseRecursively("hello"));
    }
}