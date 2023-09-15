#/bin/bash

# Change to the project directory
cd "$(dirname "$0")"/..

find . -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.c' -o -name '*.h' \) ! -path "./ggml/*" ! -path "./build/*" -exec clang-format -i {} +
