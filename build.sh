#!/bin/bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
echo "Build finished. Output: build/qt_lprnet_app"
