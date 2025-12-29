cmake_minimum_required(VERSION 4.1)
project(Cracking_Coding)

set(CMAKE_CXX_STANDARD 14)

add_executable(Cracking_Coding
        "Beginner/1. Reverse a String/reverse_string_stl.cpp"
        "Beginner/1. Reverse a String/reverse_string_iterative_swap.cpp"
        "Beginner/1. Reverse a String/reverse_string_using_recursion.cpp")
