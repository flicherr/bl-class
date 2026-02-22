cmake_minimum_required(VERSION 3.16)
project(lineur LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(COMMON_FLAGS
	-Wall
	-Wextra
	-Werror
	-pedantic
	-march=native
)

# App
add_subdirectory(app/gui)
add_subdirectory(app/core)
add_executable(${PROJECT_NAME} app/main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE
		gui
		core
)

target_compile_options(${PROJECT_NAME} PRIVATE ${COMMON_FLAGS})