#/bin/bash

# Change to the project directory
cd "$(dirname "$0")"/..

rm -rf ./build

mkdir build

cd build

cmake -DBUILD_SHARED_LIBS=ON -DCLIP_NATIVE=OFF ..

make

make install
