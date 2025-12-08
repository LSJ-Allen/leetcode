cmake_minimum_required(VERSION 3.10)
project(leetcode VERSION 1.0)

# ============================================================================
# Global Configuration Variables
# ============================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags (optional - uncomment if needed)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

# ============================================================================
# Testing
# ============================================================================
enable_testing()

# ============================================================================
# Include Directories
# ============================================================================
include_directories(${PROJECT_SOURCE_DIR}/include)

# ============================================================================
# Root main.cpp (if exists)
# ============================================================================
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
    add_executable(${PROJECT_NAME}_main "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
    add_test(NAME ${PROJECT_NAME}_main COMMAND ${PROJECT_NAME}_main)
    message(STATUS "Added root executable: ${PROJECT_NAME}_main")
endif()

# ============================================================================
# Auto-discover and add all subdirectories
# ============================================================================
file(GLOB SUBDIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} */)
foreach (SUBDIR ${SUBDIRS})
    # Skip utility directories
    if (NOT (${SUBDIR} MATCHES "^(build|cmake-build.*|include|bin|lib|\\.git|\\.vscode|\\.idea)/"))
        if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUBDIR}/CMakeLists.txt")
            add_subdirectory(${SUBDIR})
            message(STATUS "Added subdirectory: ${SUBDIR}")
        endif()
    endif()
endforeach()

# ============================================================================
# Configuration Summary
# ============================================================================
message(STATUS "========================================")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "C++ Standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "========================================")