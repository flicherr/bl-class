# GLFW
include(FetchContent)
FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.3.9
)
FetchContent_MakeAvailable(glfw)

# GLAD
add_library(glad
        ${PROJECT_SOURCE_DIR}/lib/glad/src/glad.c
)
target_include_directories(glad PUBLIC
        ${PROJECT_SOURCE_DIR}/lib/glad/include
)

# ImGui
add_library(imgui
        ${PROJECT_SOURCE_DIR}/lib/imgui/imgui.cpp
        ${PROJECT_SOURCE_DIR}/lib/imgui/imgui_draw.cpp
        ${PROJECT_SOURCE_DIR}/lib/imgui/imgui_tables.cpp
        ${PROJECT_SOURCE_DIR}/lib/imgui/imgui_widgets.cpp
        ${PROJECT_SOURCE_DIR}/lib/imgui/backends/imgui_impl_glfw.cpp
        ${PROJECT_SOURCE_DIR}/lib/imgui/backends/imgui_impl_opengl3.cpp
)

target_include_directories(imgui PUBLIC
        ${PROJECT_SOURCE_DIR}/lib/imgui
        ${PROJECT_SOURCE_DIR}/lib/imgui/backends
)

target_link_libraries(imgui PUBLIC glfw glad)

add_library(gui
        gui_manager.cpp
)
target_include_directories(gui PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/app/core
        ${PROJECT_SOURCE_DIR}/lib/glad/include
        ${PROJECT_SOURCE_DIR}/lib/imgui
)
target_link_libraries(gui INTERFACE
        core
        imgui
        glfw
        glad
)