
cmake_minimum_required(VERSION 3.11)
project(qt_lprnet_customer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt5 REQUIRED COMPONENTS Core Widgets Gui)

set(OpenCV_DIR "/opt/opencv-4.5.0/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV dir: ${OpenCV_DIR}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/src
    ${CMAKE_SOURCE_DIR}/include/eigen3
    ${OpenCV_INCLUDE_DIRS}
    ${Qt5Core_INCLUDE_DIRS}
    ${Qt5Widgets_INCLUDE_DIRS}
    ${Qt5Gui_INCLUDE_DIRS}
)

set(UI_SOURCES
    src/qt_lprnet/main.cpp
    src/qt_lprnet/mainwindow.cpp
    src/qt_lprnet/lprnet.cpp
    src/qt_lprnet/ppocr.cpp
)

set(MOC_HEADERS src/qt_lprnet/mainwindow.h)
qt5_wrap_cpp(MOC_FILES ${MOC_HEADERS})

add_executable(qt_lprnet_app ${UI_SOURCES} ${MOC_FILES})

set_target_properties(qt_lprnet_app PROPERTIES
    BUILD_RPATH "${CMAKE_SOURCE_DIR}/lib;/opt/opencv-4.5.0/lib"
    INSTALL_RPATH "$ORIGIN/../lib;/opt/opencv-4.5.0/lib"
    LINK_FLAGS "-Wl,-rpath-link,${CMAKE_SOURCE_DIR}/lib -Wl,-rpath-link,/opt/opencv-4.5.0/lib"
)

target_link_libraries(qt_lprnet_app
    Qt5::Core Qt5::Gui Qt5::Widgets

    # OpenCV 4.5.0
    ${OpenCV_LIBS}

    # 闭源库
    ${CMAKE_SOURCE_DIR}/lib/libdraw_lib.so
    ${CMAKE_SOURCE_DIR}/lib/libyolov11_lib.so
    ${CMAKE_SOURCE_DIR}/lib/libyolov11_thread_pool_lib.so

    # Rockchip 库
    ${CMAKE_SOURCE_DIR}/lib/librknnrt.so
    ${CMAKE_SOURCE_DIR}/lib/librga.so

    pthread
    dl
    m
    z
    -Wl,--no-as-needed
)