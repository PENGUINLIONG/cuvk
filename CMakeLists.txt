cmake_minimum_required(VERSION 3.5)
project(cuvk LANGUAGES CXX VERSION 0.0.1)

find_package(Vulkan REQUIRED)
find_package(PNG REQUIRED)

message("Project 'CUVK':")
message("           C++: ${CMAKE_CXX_COMPILER}")
message("        Source: ${cuvk_SOURCE_DIR}")
message("        Vulkan: ${Vulkan_INCLUDE_DIR}")
message("           PNG: ${PNG_INCLUDE_DIR}")
message(" Binary Output: ${CMAKE_CURRENT_BINARY_DIR}")
message("")

aux_source_directory(src/cuvk DIR_SRCS)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/assets/shaders/deform.comp.spv
          ${CMAKE_CURRENT_SOURCE_DIR}/assets/shaders/eval.vert.spv
          ${CMAKE_CURRENT_SOURCE_DIR}/assets/shaders/eval.geom.spv
          ${CMAKE_CURRENT_SOURCE_DIR}/assets/shaders/eval.frag.spv
          ${CMAKE_CURRENT_SOURCE_DIR}/assets/shaders/cost.comp.spv
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/assets/shaders)

add_subdirectory(third/fmt)

add_definitions(-DL_CUVK_COMPILE)

# CUVK shared library.
add_library(libcuvk SHARED ${DIR_SRCS})
set_target_properties(libcuvk PROPERTIES PUBLIC_HEADER include/cuvk.h)
target_include_directories(libcuvk PRIVATE include
                                           ${Vulkan_INCLUDE_DIR}
                                           ${PNG_INCLUDE_DIR})
target_link_libraries(libcuvk PRIVATE fmt-header-only
                                      ${Vulkan_LIBRARIES}
                                      ${PNG_LIBRARIES})
set_property(TARGET libcuvk PROPERTY CXX_STANDARD 17)
set_property(TARGET libcuvk PROPERTY CXX_STANDARD_REQUIRED ON)
