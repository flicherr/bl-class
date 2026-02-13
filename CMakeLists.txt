cmake_minimum_required(VERSION 3.16)
project(bl-class LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(COMMON_FLAGS
	-Wall
	-Wextra
	-Werror
	-pedantic
	-march=native
)

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
	lib/glad/src/glad.c
)
target_include_directories(glad PUBLIC
	lib/glad/include
)


# ImGui
add_library(imgui
	lib/imgui/imgui.cpp
	lib/imgui/imgui_draw.cpp
	lib/imgui/imgui_tables.cpp
	lib/imgui/imgui_widgets.cpp
	lib/imgui/backends/imgui_impl_glfw.cpp
	lib/imgui/backends/imgui_impl_opengl3.cpp
)

target_include_directories(imgui PUBLIC
	lib/imgui
	lib/imgui/backends
)

target_link_libraries(imgui PUBLIC glfw glad)


# App
add_executable(${PROJECT_NAME}
		app/main.cpp
		app/core/perceptron_threshold.cpp
		app/core/perceptron_bias.cpp
		app/core/trainer.cpp
		app/core/dataset.cpp
)

target_include_directories(${PROJECT_NAME}
	PRIVATE
		app/core
		lib/glad/include
		lib/imgui
)

target_link_libraries(${PROJECT_NAME}
	imgui
	glfw
	glad
)

target_compile_options(${PROJECT_NAME} PRIVATE ${COMMON_FLAGS})
