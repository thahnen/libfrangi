cmake_minimum_required(VERSION 3.14)
project(libfrangi
        VERSION         1.0.0
        DESCRIPTION     ""
        HOMEPAGE_URL    ""
        LANGUAGES       CXX)

set(CMAKE_BUILD_TYPE Release)

# OpenMP required for parallized working
find_package(OpenMP REQUIRED)
# OpenCV required for image processing
find_package(OpenCV REQUIRED)

include_directories(./include)

#compile
add_library(frangi SHARED src/frangi.cpp)
add_executable(frangi-bin example/main.cpp)
target_link_libraries(frangi-bin frangi opencv_core opencv_imgproc opencv_highgui)

#install
install (TARGETS frangi DESTINATION lib)
install (TARGETS frangi-bin DESTINATION bin)
install (FILES include/frangi.h DESTINATION include)
