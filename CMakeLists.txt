cmake_minimum_required(VERSION 3.14)
project(frangi
        VERSION         1.5.5
        DESCRIPTION     "Implementation of a standard vesselness filter"
        HOMEPAGE_URL    "https://github.com/thahnen/libfrangi"
        LANGUAGES       CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)


# OpenMP required for parallelized working
find_package(OpenMP REQUIRED)

# OpenCV required for image processing
find_package(OpenCV REQUIRED)


include_directories(${PROJECT_SOURCE_DIR}/include)


########################################################################################################################
#
#       SHARED LIBRARY (using OpenMP for parallelization)
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


########################################################################################################################
#       AFTER LIBRARIES ARE BUILD FIND THEM!
########################################################################################################################
find_library(frangi REQUIRED)
find_library(frangiNP REQUIRED)


########################################################################################################################
#
#       BUILD OPTIONS FOR BENCHMARKING:
#       => compares the libraries with and without OpenMP
#
########################################################################################################################
add_executable(frangi_test_with
        example/main.cpp)

target_link_libraries(frangi_test_with
        PUBLIC
            frangi)

add_executable(frangi_test_without
        example/main.cpp)

target_link_libraries(frangi_test_without
        PUBLIC
            frangiNP)


#install
#install (TARGETS frangi DESTINATION lib)
#install (FILES include/frangi.h DESTINATION include)
