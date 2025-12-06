cmake_minimum_required(VERSION 3.10)
project(leetcode VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable testing
enable_testing()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add the root executable (from root main.cpp)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
    add_executable(${PROJECT_NAME}_main "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
    add_test(NAME ${PROJECT_NAME}_main COMMAND ${PROJECT_NAME}_main)
    message(STATUS "Added root executable: ${PROJECT_NAME}_main")
endif()

# Get all problem subdirectories
file(GLOB PROBLEM_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} */)
foreach(PROBLEM_DIR ${PROBLEM_DIRS})
    # Skip utility directories
    if(NOT (${PROBLEM_DIR} MATCHES "^(build|include|bin|lib|.git|.vscode)/"))
        # Check if this problem directory has its own CMakeLists.txt
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${PROBLEM_DIR}/CMakeLists.txt")
            # Use the subdirectory's own CMakeLists.txt
            add_subdirectory(${PROBLEM_DIR})
            message(STATUS "Added problem with custom CMake: ${PROBLEM_DIR}")
            # Otherwise, if it has a main.cpp, create an executable for it
        elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${PROBLEM_DIR}/main.cpp")
            # Extract problem name from directory path
            get_filename_component(PROBLEM_NAME ${PROBLEM_DIR} NAME)

            # Create executable for this problem using ONLY its main.cpp
            # This avoids the "multiple definition of main" error
            add_executable(${PROBLEM_NAME} "${CMAKE_CURRENT_SOURCE_DIR}/${PROBLEM_DIR}/main.cpp")

            # Find other non-main source files in this problem directory
            file(GLOB_RECURSE PROBLEM_SOURCES
                    "${PROBLEM_DIR}/*.cpp"
                    "${PROBLEM_DIR}/*.h"
                    "${PROBLEM_DIR}/*.hpp"
            )
            # Filter out main.cpp from the sources
            list(FILTER PROBLEM_SOURCES EXCLUDE REGEX ".*main\\.cpp$")

            # Add the other source files to the executable
            if(PROBLEM_SOURCES)
                target_sources(${PROBLEM_NAME} PRIVATE ${PROBLEM_SOURCES})
            endif()

            # Add test
            add_test(NAME ${PROBLEM_NAME} COMMAND ${PROBLEM_NAME})

            message(STATUS "Added problem: ${PROBLEM_NAME}")
        endif()
    endif()
endforeach()

# Print configuration information
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")