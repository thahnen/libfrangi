cmake_minimum_required(VERSION 3.14)
project(frangi
        VERSION         1.0.0
        DESCRIPTION     "Implementation of a standard vesselness filter"
        HOMEPAGE_URL    "https://github.com/thahnen/libfrangi"
        LANGUAGES       CXX)

set(CMAKE_BUILD_TYPE Release)

# OpenMP required for parallized working
find_package(OpenMP REQUIRED)
# OpenCV required for image processing
find_package(OpenCV REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)


########################################################################################################################
#
#       SHARED LIBRARY (using OpenMP for parallized working)
#
########################################################################################################################
add_library(${PROJECT_NAME}
        SHARED
            src/frangi.cpp)

set_target_properties(${PROJECT_NAME}
        PROPERTIES
            VERSION ${PROJECT_VERSION}
            CXX_VISIBILITY_PRESET hidden
            PUBLIC_HEADER include/frangi.h)

target_include_directories(${PROJECT_NAME}
        PUBLIC
            ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(${PROJECT_NAME}
        OpenMP::OpenMP_CXX
        ${OpenCV_LIBS})


########################################################################################################################
#
#       SHARED LIBRARY (not using OpenMP)
#
########################################################################################################################
add_library(frangiNP
        SHARED
            src/frangi.cpp)

set_target_properties(frangiNP
        PROPERTIES
            VERSION ${PROJECT_VERSION}
            CXX_VISIBILITY_PRESET hidden
            PUBLIC_HEADER include/frangi.h)

target_include_directories(frangiNP
        PUBLIC
            ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(frangiNP
        ${OpenCV_LIBS})

target_compile_definitions(frangiNP
        PUBLIC
            NO_OMP)



#compile
#add_library(frangi SHARED src/frangi.cpp)
add_executable(frangi-bin example/main.cpp)
target_link_libraries(frangi-bin frangi opencv_core opencv_imgproc opencv_highgui)

#install
#install (TARGETS frangi DESTINATION lib)
#install (TARGETS frangi-bin DESTINATION bin)
install (FILES include/frangi.h DESTINATION include)
