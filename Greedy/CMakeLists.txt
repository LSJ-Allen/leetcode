# CMakeLists.txt for a specific LeetCode problem subfolder (e.g., two-sum)

# Set minimum required CMake version
cmake_minimum_required(VERSION 3.10)

# Get the directory name as the problem name
get_filename_component(PROBLEM_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Add project command
project(${PROBLEM_NAME})

# Create executable for this problem using ONLY main.cpp initially
add_executable(${PROBLEM_NAME} "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")

# Find all source files EXCEPT main.cpp
# This more explicit approach avoids including any main.cpp files
file(GLOB ADDITIONAL_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
)

# Manually exclude main.cpp from the sources
# This is a more reliable way to ensure main.cpp isn't added twice
foreach(SOURCE_FILE ${ADDITIONAL_SOURCES})
    get_filename_component(FILE_NAME ${SOURCE_FILE} NAME)
    if(NOT ${FILE_NAME} STREQUAL "main.cpp")
        target_sources(${PROBLEM_NAME} PRIVATE ${SOURCE_FILE})
    endif()
endforeach()

# Add test for this problem
add_test(NAME ${PROBLEM_NAME}_test COMMAND ${PROBLEM_NAME})

message(STATUS "Configured problem: ${PROBLEM_NAME} (main.cpp + ${ADDITIONAL_SOURCES_COUNT} additional files)")