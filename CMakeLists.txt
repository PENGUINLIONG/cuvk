cmake_minimum_required(VERSION 3.5)
project(cuvk LANGUAGES CXX VERSION 0.0.1)

find_package(vulkan REQUIRED)
find_package(png REQUIRED)

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
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/assets/shaders)

add_executable(cuvk ${DIR_SRCS})
target_include_directories(cuvk PRIVATE include
                                        ${Vulkan_INCLUDE_DIR}
                                        ${PNG_INCLUDE_DIR})

add_subdirectory(third/fmt)
target_link_libraries(cuvk PRIVATE fmt-header-only
                                   ${Vulkan_LIBRARIES}
                                   ${PNG_LIBRARIES})

set_property(TARGET cuvk PROPERTY CXX_STANDARD 17)
set_property(TARGET cuvk PROPERTY CXX_STANDARD_REQUIRED ON)
