#/bin/bash
find . -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.c' -o -name '*.h' \) ! -path "./ggml/*" ! -path "./build/*" -exec clang-format -i {} +
