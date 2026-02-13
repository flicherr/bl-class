#include <iostream>
#include <memory>
#include "core/dataset.h"
#include "core/perceptron.h"
#include "core/trainer.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>

// FIXME: transfer the logic of the visualization interface from main.cpp in the core/gui
struct AppState
{
	TypePreceptron type_preceptron;
	std::vector<Sample> train_data;
	std::vector<Sample> test_data;
	std::vector<Weights> history;
	int iteration = 0;
	Weights current_weights{0, 0, 1};

	bool isPlaying = false;
	float playSpeed = 1.0f;
	double lastUpdateTime = 0.0;
};

ImVec2 worldToScreen(double x, double y,
                     ImVec2 origin, ImVec2 size, double scale)
{
    return {
        origin.x + size.x * 0.5f + (float)(x * scale),
        origin.y + size.y * 0.5f - (float)(y * scale)
    };
}

void drawCanvas(AppState &state)
{
	ImGui::Begin("Canvas");

	ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
	ImVec2 canvas_size = ImGui::GetContentRegionAvail();
	if (canvas_size.x < 50) canvas_size.x = 50;
	if (canvas_size.y < 50) canvas_size.y = 50;

	ImDrawList* draw_list = ImGui::GetWindowDrawList();

	draw_list->AddRectFilled(
		canvas_pos,
		ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y),
		IM_COL32(40, 40, 40, 255)
	);

	draw_list->AddRect(
		canvas_pos,
		ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y),
		IM_COL32(255, 255, 255, 255)
	);

	ImGui::InvisibleButton("canvas", canvas_size);

	double scale = 75.0;

	for (const auto &s : state.train_data) {
		ImVec2 p = worldToScreen(s.x1, s.x2, canvas_pos, canvas_size, scale);
		ImU32 color = (s.y == 1) ? IM_COL32(255, 80, 80, 255)
								  : IM_COL32(80, 160, 255, 255);
		draw_list->AddCircleFilled(p, 4.0f, color);
	}
	for (const auto &s : state.test_data) {
		ImVec2 p = worldToScreen(s.x1, s.x2, canvas_pos, canvas_size, scale);
		ImU32 color = (s.y == 1) ? IM_COL32(255, 20, 20, 255)
								  : IM_COL32(20, 100, 255, 255);
		draw_list->AddCircleFilled(p, 4.0f, color);
	}


	const Weights& w = state.current_weights;
	if (state.type_preceptron == Bias) {
		if (w.w2 != 0.0) {
			double x_min = -10.0, x_max = 10.0;
			double y1 = -(w.w1 * x_min + w.w0) / w.w2;
			double y2 = -(w.w1 * x_max + w.w0) / w.w2;
			ImVec2 p1 = worldToScreen(x_min, y1, canvas_pos, canvas_size, scale);
			ImVec2 p2 = worldToScreen(x_max, y2, canvas_pos, canvas_size, scale);
			draw_list->AddLine(p1, p2, IM_COL32(200, 200, 200, 255), 2.0f);
		} else if (w.w1 != 0.0) {
			double x = -w.w0 / w.w1;
			ImVec2 p1 = worldToScreen(x, -10, canvas_pos, canvas_size, scale);
			ImVec2 p2 = worldToScreen(x,  10, canvas_pos, canvas_size, scale);
			draw_list->AddLine(p1, p2, IM_COL32(200, 200, 200, 255), 2.0f);
		}
	} else {
		if (w.w2 != 0.0) {
			double x_min = -10.0, x_max = 10.0;
			double y1 = -(w.w1 * x_min + w.w0) / w.w2;
			double y2 = -(w.w1 * x_max + w.w0) / w.w2;
			ImVec2 p1 = worldToScreen(x_min, y1, canvas_pos, canvas_size, scale);
			ImVec2 p2 = worldToScreen(x_max, y2, canvas_pos, canvas_size, scale);
			draw_list->AddLine(p1, p2, IM_COL32(200, 200, 200, 255), 2.0f);
		} else if (w.w1 != 0.0) {
			double x = w.w0 / w.w1;
			ImVec2 p1 = worldToScreen(x, -10, canvas_pos, canvas_size, scale);
			ImVec2 p2 = worldToScreen(x,  10, canvas_pos, canvas_size, scale);
			draw_list->AddLine(p1, p2, IM_COL32(200, 200, 200, 255), 2.0f);
		}
	}

	ImGui::End();
}

int main()
{
	glfwSetErrorCallback([](int error, const char* desc){
		std::cerr << "GLFW Error " << error << ": " << desc << "\n";
	});

	if (!glfwInit()) return -1;

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(900, 680, "bl-class", nullptr, nullptr);
	if (!window) return -1;
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	if (!gladLoadGL()) {
		std::cerr << "Failed to initialize GLAD\n";
		return -1;
	}

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiStyle& style = ImGui::GetStyle();

	style.WindowRounding = 8.0f;
	style.FrameRounding  = 6.0f;
	style.ScrollbarRounding = 6.0f;
	style.FrameBorderSize  = 1.0f;
	style.WindowBorderSize = 1.0f;
	style.WindowPadding = ImVec2(12, 12);
	style.FramePadding  = ImVec2(8, 4);
	style.ItemSpacing   = ImVec2(8, 6);

	style.ScaleAllSizes(1.0f);

	ImGuiIO& io = ImGui::GetIO(); (void)io;
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330");

	try	{
		auto data = Dataset::load_from_file("../scripts/train.csv");
		auto split = Dataset::split(data);

		AppState state;
		state.train_data = split.train;
		state.test_data = split.test;

		std::unique_ptr<IPerceptron> model;
		model = std::make_unique<PerceptronBias>();
		state.type_preceptron = Bias;

		Trainer trainer;
		trainer.train(*model, split.train, 100);

		state.history = trainer.history;

		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();

			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();

			ImGui::Begin("Controls");

			if (state.isPlaying) {
				if (ImGui::Button("Pause")) state.isPlaying = false;
			} else {
				if (ImGui::Button("Play")) state.isPlaying = true;
			}

			ImGui::SliderInt("Iteration", &state.iteration, 0, (int)state.history.size()-1);
			ImGui::SliderFloat("Speed (iters/sec)", &state.playSpeed, 0.1f, 10.0f);

			ImGui::End();

			if (state.isPlaying) {
				double currentTime = glfwGetTime();
				if (currentTime - state.lastUpdateTime >= 1.0 / state.playSpeed) {
					state.iteration++;
					if (state.iteration >= (int)state.history.size()) {
						state.iteration = (int)state.history.size() - 1;
						state.isPlaying = false;
					}
					state.lastUpdateTime = currentTime;
				}
			}
			state.current_weights = state.history[state.iteration];

			drawCanvas(state);
			ImGui::Render();
			int display_w, display_h;
			glfwGetFramebufferSize(window, &display_w, &display_h);
			glViewport(0, 0, display_w, display_h);
			glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT);

			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
			glfwSwapBuffers(window);
		}
	}
    catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << '\n';
    }

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	glfwDestroyWindow(window);
	glfwTerminate();

    return 0;
}