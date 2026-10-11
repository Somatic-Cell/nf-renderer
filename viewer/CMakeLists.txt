set(project_SRC
    glfw_window.cpp
)

set(project_HEADERS
    glfw_window.hpp
)

add_library(viewer STATIC
    ${project_SRC}
    ${project_HEADERS}
)

# This target has UTF-8 source comments. Options on the renderer executable
# (or in its parent directory after add_subdirectory) do not apply here.
target_compile_options(viewer PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/utf-8>
)

target_include_directories(viewer
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
