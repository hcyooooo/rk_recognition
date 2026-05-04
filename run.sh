#!/bin/bash
set -e
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=$PWD/lib:/opt/opencv-4.5.0/lib:$LD_LIBRARY_PATH
./build/qt_lprnet_app