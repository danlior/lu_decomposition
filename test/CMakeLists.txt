# lu_test target
add_executable(lu_test lu_test.cpp)
#target_compile_options(lu_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(lu_tester lu_test)


