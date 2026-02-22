#include <iostream>
#include <cmath>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <glad/glad.h>

#include "gui_manager.h"
#include "appstate.h"

#include "trainers/trainers.h"
#include "dataset.h"

gui::Manager::~Manager()
{
    cleanup();
}

bool gui::Manager::init()
{
    if (!init_glfw()) {
        return false;
    }
    init_imgui();
    return true;
}

bool gui::Manager::init_glfw()
{
    glfwSetErrorCallback([](int error, const char* desc){
        std::cerr << "GLFW Error " << error << ": " << desc << "\n";
    });

    if (!glfwInit()) return false;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    _window = glfwCreateWindow(1000, 780, "lineur", nullptr, nullptr);
    if (!_window) return false;
    glfwMakeContextCurrent(_window);
    glfwSwapInterval(1);

    if (!gladLoadGL()) {
        std::cerr << "Failed to initialize GLAD\n";
        return false;
    }

    return true;
}

void gui::Manager::init_imgui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiStyle &style = ImGui::GetStyle();

    style.WindowRounding = 8.0f;
    style.FrameRounding  = 6.0f;
    style.ScrollbarRounding = 6.0f;
    style.FrameBorderSize  = 1.0f;
    style.WindowBorderSize = 1.0f;
    style.WindowPadding = ImVec2(12, 12);
    style.FramePadding  = ImVec2(8, 4);
    style.ItemSpacing   = ImVec2(8, 6);
    style.ScaleAllSizes(1.0f);

    ImGuiIO &io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(_window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
}


void gui::Manager::cleanup()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(_window);
    glfwTerminate();
}

void gui::Manager::drawing()
{
    try {
        set_appstate();
        while (!glfwWindowShouldClose(_window)) {
            glfwPollEvents();

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            draw_toolbar();
            draw_canvas();

            ImGui::Render();
            int display_w, display_h;
            glfwGetFramebufferSize(_window, &display_w, &display_h);
            glViewport(0, 0, display_w, display_h);
            glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(_window);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';

    }
}

void gui::Manager::set_appstate()
{
	_state = std::make_shared<AppState>();
	{
		std::vector<RegSample> data = dataset::load_from_file<RegSample>(
									  "../scripts/regression/train.csv");
		auto split = dataset::split(data);

		_state->reg_data.train = split.train;
		_state->reg_data.test = split.test;

		neurons::Regression model;
		RegressionTrainer trainer;
		trainer.train(model, split.train, 100);
		_state->reg_data.history = trainer.history;
	}
	{
		std::vector<ClSample> data = dataset::load_from_file<ClSample>(
									"../scripts/classification/train.csv");
		dataset::DatasetSplit<ClSample> split = dataset::split<ClSample>(data);

		_state->class_data.train = split.train;
		_state->class_data.test = split.test;

		ClassifierTrainer trainer;
		std::unique_ptr<neurons::IClassifier> model = std::make_unique<neurons::ClassifierByThreshold>();
		_state->class_data.type_class[neurons::Threshold] = neurons::TypeClassifier::Threshold;

		trainer.train(*model, split.train, 100);
		_state->class_data.history[neurons::Threshold] = trainer.history;
		model.release();

		model = std::make_unique<neurons::ClassifierByBias>();
		_state->class_data.type_class[neurons::Bias] = neurons::TypeClassifier::Bias;

		trainer.train(*model, split.train, 100);
		_state->class_data.history[neurons::Bias] = trainer.history;

		_state->class_data.current_weights[neurons::Threshold]
		= {0.0f, 1.0f, 0.0f};
		_state->class_data.current_weights[neurons::Bias]
		= {0.0f, 1.0f, 0.0f};
	}
}

ImVec2 gui::Manager::worldToScreen(double x, double y,ImVec2 origin, ImVec2 size) const
{
    return {
        origin.x + size.x * 0.5f + static_cast<float>(x * _state->scale) + _state->offset.x,
        origin.y + size.y * 0.5f - static_cast<float>(y * _state->scale) + _state->offset.y
    };
}

void gui::Manager::draw_canvas() const
{
    ImGui::Begin("Visualization");

	ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
	ImVec2 canvas_size = ImGui::GetContentRegionAvail();
	if (canvas_size.x < 50) canvas_size.x = 50;
	if (canvas_size.y < 50) canvas_size.y = 50;

	ImDrawList *draw_list = ImGui::GetWindowDrawList();

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

	for (float x = fmodf(canvas_pos.x, 30); x < canvas_size.x; x += 30) {
		draw_list->AddLine({canvas_pos.x + x, canvas_pos.y},
						   {canvas_pos.x + x, canvas_pos.y + canvas_size.y - 2},
						   IM_COL32(60,60,60,80));
	}

	for (float y = fmodf(canvas_pos.y, 30); y < canvas_size.y; y += 30) {
		draw_list->AddLine({canvas_pos.x, canvas_pos.y + y},
						   {canvas_pos.x + canvas_size.x, canvas_pos.y + y - 2},
						   IM_COL32(60,60,60,80));
	}

	ImGui::InvisibleButton("canvas", canvas_size);
	const ImGuiIO &io = ImGui::GetIO();
	if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
		ImVec2 drag_delta = io.MouseDelta;
		_state->offset.x += drag_delta.x;
		_state->offset.y += drag_delta.y;
	}

	if (ImGui::IsItemHovered()) {
		constexpr float zoomSpeed = 0.1f;
		_state->scale *= (1.0f + io.MouseWheel * zoomSpeed);
	}

	if (!_state->currentShownData) { // CurrentShownData::Regression = 0 => false
		// train points
		for (const auto & [x, y] : _state->reg_data.train) {
			ImVec2 p = worldToScreen(x, y, canvas_pos, canvas_size);
			draw_list->AddCircleFilled(p, 3.5f, IM_COL32(80, 80, 255, 255));
		}
		// test points
		for (const auto & [x, y] : _state->reg_data.test) {
			ImVec2 p = worldToScreen(x, y, canvas_pos, canvas_size);
			draw_list->AddCircleFilled(p, 3.5f, IM_COL32(200, 200, 80, 255));
		}
	} else {
		// train points
		ImVec2 p;
		ImU32 color;
		for (const auto & [x1, x2, y] : _state->class_data.train) {
			p = worldToScreen(x1, x2, canvas_pos, canvas_size);
			color = (y == 1)  ? IM_COL32(255, 80, 80, 255)
								: IM_COL32(80, 160, 255, 255);
			draw_list->AddCircleFilled(p, 3.5f, color);
		}
		// test points
		for (const auto & [x1, x2, y] : _state->class_data.test) {
			p = worldToScreen(x1, x2, canvas_pos, canvas_size);
			color = (y == 1)	? IM_COL32(255, 80, 80, 255)
								: IM_COL32(80, 160, 255, 255);
			draw_list->AddCircleFilled(p, 3.5f, color);
		}
	}

	// line
    constexpr double x_min = -20000.0;
	constexpr double x_max =  20000.0;

	ImVec2 p1, p2;
	double y1, y2;

	if (!_state->currentShownData) { // CurrentShownData::Regression = 0 => false
		const auto & [w1, w0] = _state->reg_data.current_weights;
		y1 = w1 * x_min + w0;
		y2 = w1 * x_max + w0;
		p1 = worldToScreen(x_min, y1, canvas_pos, canvas_size);
		p2 = worldToScreen(x_max, y2, canvas_pos, canvas_size);
	} else {
		if (_state->currentShownData == AppState::ClassificationBias) {
			if (const auto & [w1, w2, w0] = _state->class_data
				.current_weights[neurons::Bias]; w2 != 0.0) {
				y1 = -(w1 * x_min + w0) / w2;
				y2 = -(w1 * x_max + w0) / w2;
				p1 = worldToScreen(x_min, y1, canvas_pos, canvas_size);
				p2 = worldToScreen(x_max, y2, canvas_pos, canvas_size);
			} else if (w1 != 0.0) {
				double x = -w0 / w1;
				p1 = worldToScreen(x, -10, canvas_pos, canvas_size);
				p2 = worldToScreen(x,  10, canvas_pos, canvas_size);
			}
		} else if (_state->currentShownData == AppState::ClassificationThreshold) {
			if (const auto & [w1, w2, w0] = _state->class_data
				.current_weights[neurons::Threshold]; w2 != 0.0) {
				y1 = -(w1 * x_min + w0) / w2;
				y2 = -(w1 * x_max + w0) / w2;
				p1 = worldToScreen(x_min, y1, canvas_pos, canvas_size);
				p2 = worldToScreen(x_max, y2, canvas_pos, canvas_size);
			} else if (w1 != 0.0) {
				const double x = w0 / w1;
				p1 = worldToScreen(x, -10, canvas_pos, canvas_size);
				p2 = worldToScreen(x,  10, canvas_pos, canvas_size);
			}
		}
	}

	draw_list->AddLine(p1, p2, IM_COL32(220, 220, 220, 255), 2.5f);
	ImGui::End();
}

void gui::Manager::draw_toolbar() const
{
	ImGui::Begin("Controls", nullptr,
				 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar);
	const float avail = ImGui::GetContentRegionAvail().x;

	ImGui::BeginGroup();
	if (_state->isPlaying) {
		if (ImGui::Button("Pause")) _state->isPlaying = false;
	} else {
		if (ImGui::Button("Play")) _state->isPlaying = true;
	}
	ImGui::EndGroup();

	ImGui::SameLine(70.0f);

	ImGui::BeginGroup();
	if (ImGui::Button("Reset")) {
		_state->iteration = 0;
		_state->isPlaying = false;
	}
	ImGui::EndGroup();

	constexpr float btn_w = 210.0f;
	ImGui::SameLine(avail - btn_w);

	ImGui::SetNextItemWidth(220.0f);
	ImGui::BeginGroup();
	const char *items[] = {
		"Regression",
		"Classification (t)",
		"Classification (w0)",
	};

	int idx = _state->currentShownData;
	if (ImGui::Combo("##combo_types", &idx, items, IM_ARRAYSIZE(items))) {
		_state->currentShownData = static_cast<AppState::CurrentShownData>(idx);
		_state->iteration = 0;
		_state->isPlaying = false;
		_state->lastUpdateTime = 0.0;
	}
	ImGui::EndGroup();

	int iterations = 0;
	if (!_state->currentShownData) {
		iterations = _state->reg_data.history.size();
	} else {
		iterations = _state->class_data.history.size();
	}
	ImGui::SliderInt("Iteration", &_state->iteration, 0, iterations - 1);
	ImGui::SliderFloat("Speed (iters/sec)", &_state->playSpeed, 0.1f, 10.0f);

	constexpr float separator = 110.0f;
	if (_state->currentShownData == AppState::Regression) {
		ImGui::BeginGroup();
		ImGui::Text("w1 (a) = %.4f;", _state->reg_data.current_weights.w1);
		ImGui::EndGroup();
		ImGui::SameLine(separator * 1.3f);
		ImGui::BeginGroup();
		ImGui::Text("w0 (b) = %.4f", _state->reg_data.current_weights.w0);
		ImGui::EndGroup();
	} else if (_state->currentShownData == AppState::ClassificationThreshold) {
		ImGui::BeginGroup();
		ImGui::Text("w1 = %.4f;", _state->class_data.current_weights[neurons::Threshold].w1);
		ImGui::EndGroup();
		ImGui::SameLine(separator);
		ImGui::BeginGroup();
		ImGui::Text(" w2 = %.4f;", _state->class_data.current_weights[neurons::Threshold].w2);
		ImGui::EndGroup();
		ImGui::SameLine(2 * separator);
		ImGui::BeginGroup();
		ImGui::Text("t = %.4f ", _state->class_data.current_weights[neurons::Threshold].w0);
		ImGui::EndGroup();
	} else if (_state->currentShownData == AppState::ClassificationBias) {
		ImGui::BeginGroup();
		ImGui::Text("w1 = %.4f;", _state->class_data.current_weights[neurons::Bias].w1);
		ImGui::EndGroup();
		ImGui::SameLine(separator);
		ImGui::BeginGroup();
		ImGui::Text(" w2 = %.4f;", _state->class_data.current_weights[neurons::Bias].w2);
		ImGui::EndGroup();
		ImGui::SameLine(2 * separator);
		ImGui::BeginGroup();
		ImGui::Text("w0 = %.4f", _state->class_data.current_weights[neurons::Bias].w0);
		ImGui::EndGroup();
	}
	ImGui::End();

	if (_state->isPlaying) {
		if (const double currentTime = glfwGetTime();
			currentTime - _state->lastUpdateTime >= 1.0 / _state->playSpeed) {
			_state->iteration++;
			if (_state->iteration >= _state->iteration) {
				_state->iteration = iterations - 1;
				_state->isPlaying = false;
			}
			_state->lastUpdateTime = currentTime;
			}
	}

	if (!_state->currentShownData) {
		_state->reg_data.current_weights = _state->reg_data.history[_state->iteration];
	} else if (_state->currentShownData == AppState::ClassificationThreshold) {
		_state->class_data.current_weights[neurons::Threshold]
		= _state->class_data.history[neurons::Threshold][_state->iteration];
	} else if (_state->currentShownData == AppState::ClassificationBias) {
		_state->class_data.current_weights[neurons::Bias]
		= _state->class_data.history[neurons::Bias][_state->iteration];
	}
}