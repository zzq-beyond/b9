cmake_minimum_required(VERSION 3.16)

project(glass_surface_detection VERSION 1.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(OpenCV 4.4 REQUIRED)
find_package(Boost 1.77 COMPONENTS filesystem json REQUIRED)
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

# OpenMP for multiprocessing
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Add debug option
option(DEBUG "Debug option" OFF)
if (DEBUG)
    add_definitions(-DDEBUG)
endif()

add_library(
    ${PROJECT_NAME} SHARED 
    src/cpp/glass_surface_detection.cpp
)
target_link_libraries(
    ${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    /opt/intel/openvino_2022/runtime/lib/intel64/libopenvino.so
)

add_executable(
    test_${PROJECT_NAME} 
    src/cpp/main.cpp 
    src/cpp/glass_surface_detection.cpp
)
target_link_libraries(
    test_${PROJECT_NAME}
    ${OpenCV_LIBS} 
    ${Boost_LIBRARIES}
    /opt/intel/openvino_2022/runtime/lib/intel64/libopenvino.so
)
file(
    COPY 
    "./config.json" "./images" "./models" 
    DESTINATION ${CMAKE_BINARY_DIR}/bin
)
