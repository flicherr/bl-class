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
    switch (_state->currentShownData) {
    case AppState::ClassificationThreshold:
	    {
		    std::vector<ClSample> cldata =
		    	dataset::load_from_file<ClSample>("../scripts/classification/train.csv");
    		dataset::DatasetSplit<ClSample> clsplit = dataset::split<ClSample>(cldata);

    		_state->class_data.train = clsplit.train;
    		_state->class_data.test = clsplit.test;

    		std::unique_ptr<neurons::IClassifier> clmodel;
    		clmodel = std::make_unique<neurons::ClassifierByThreshold>();
    		_state->class_data.type_preceptron = neurons::TypeClassifier::Threshold;

    		ClassifierTrainer cltrainer;
    		cltrainer.train(*clmodel, clsplit.train, 100);
    		_state->class_data.history = cltrainer.history;
    		break;
	    }
    case AppState::ClassificationBias:
	    {
    		std::vector<ClSample> cldata =
				dataset::load_from_file<ClSample>("../scripts/classification/train.csv");
    		dataset::DatasetSplit<ClSample> clsplit = dataset::split<ClSample>(cldata);

    		_state->class_data.train = clsplit.train;
    		_state->class_data.test = clsplit.test;

    		std::unique_ptr<neurons::IClassifier> clmodel;
    		clmodel = std::make_unique<neurons::ClassifierByBias>();
    		_state->class_data.type_preceptron = neurons::TypeClassifier::Bias;

    		ClassifierTrainer cltrainer;
    		cltrainer.train(*clmodel, clsplit.train, 100);
    		_state->class_data.history = cltrainer.history;
    		break;
	    }
    case AppState::Regression:
	    {
		    std::vector<RegSample> data =
				dataset::load_from_file<RegSample>("../scripts/regression/train.csv");
    		auto split = dataset::split(data);

    		_state->reg_data.train = split.train;
    		_state->reg_data.test = split.test;

    		neurons::Regression model;
    		RegressionTrainer trainer;
    		trainer.train(model, split.train, 100);
    		_state->reg_data.history = trainer.history;
    		break;
	    }
    }
}


ImVec2 gui::Manager::worldToScreen(double x, double y,ImVec2 origin, ImVec2 size) const
{
    return {
        origin.x + size.x * 0.5f + static_cast<float>(x * _state->scale) + _state->offset.x,
        origin.y + size.y * 0.5f - static_cast<float>(y * _state->scale) + _state->offset.y
    };
}

void gui::Manager::draw_canvas()
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

	for (float x = fmodf(canvas_pos.x, 30); x < canvas_size.x; x += 30)
		draw_list->AddLine({canvas_pos.x + x, canvas_pos.y},
						   {canvas_pos.x + x, canvas_pos.y + canvas_size.y - 2},
						   IM_COL32(60,60,60,80));

	for (float y = fmodf(canvas_pos.y, 30); y < canvas_size.y; y += 30)
		draw_list->AddLine({canvas_pos.x, canvas_pos.y + y},
						   {canvas_pos.x + canvas_size.x, canvas_pos.y + y - 2},
						   IM_COL32(60,60,60,80));

	ImGui::InvisibleButton("canvas", canvas_size);
	ImGuiIO& io = ImGui::GetIO();
	if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
		ImVec2 drag_delta = io.MouseDelta;
		_state->offset.x += drag_delta.x;
		_state->offset.y += drag_delta.y;
	}

	if (ImGui::IsItemHovered()) {
		float zoomSpeed = 0.1f;
		_state->scale *= (1.0f + io.MouseWheel * zoomSpeed);
	}


	// train points
	for (const auto &s : _state->reg_data.train) {
		ImVec2 p = worldToScreen(s.x, s.y, canvas_pos, canvas_size);
		draw_list->AddCircleFilled(p, 3.5f, IM_COL32(80, 80, 255, 255));
	}

	// test points
	for (const auto &s : _state->reg_data.test) {
		ImVec2 p = worldToScreen(s.x, s.y, canvas_pos, canvas_size);
		draw_list->AddCircleFilled(p, 3.5f, IM_COL32(200, 200, 80, 255));
	}

	// regression line
	const auto &w = _state->reg_data.current_weights;

	double x_min = -20000.0;
	double x_max =  20000.0;

	double y1 = w.w1 * x_min + w.w0;
	double y2 = w.w1 * x_max + w.w0;

	ImVec2 p1 = worldToScreen(x_min, y1, canvas_pos, canvas_size);
	ImVec2 p2 = worldToScreen(x_max, y2, canvas_pos, canvas_size);

	draw_list->AddLine(p1, p2, IM_COL32(220, 220, 220, 255), 2.5f);

	ImGui::End();
}

void gui::Manager::draw_toolbar()
{
	ImGui::Begin(
				"Controls",
				nullptr,
				ImGuiWindowFlags_NoTitleBar |
				ImGuiWindowFlags_NoScrollbar
	);

	float avail = ImGui::GetContentRegionAvail().x;

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

	float btn_w = 210.0f;
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
	}
	ImGui::EndGroup();

	ImGui::SliderInt("Iteration", &_state->iteration, 0, (int)_state->reg_data.history.size()-1);
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
		ImGui::Text("w1 = %.4f;", _state->class_data.current_weights.w1);
		ImGui::EndGroup();
		ImGui::SameLine(separator);
		ImGui::BeginGroup();
		ImGui::Text(" w2 = %.4f;", _state->class_data.current_weights.w2);
		ImGui::EndGroup();
		ImGui::SameLine(2 * separator);
		ImGui::BeginGroup();
		ImGui::Text("t = %.4f ", _state->class_data.current_weights.w0);
		ImGui::EndGroup();
	} else if (_state->currentShownData == AppState::ClassificationBias) {
		ImGui::BeginGroup();
		ImGui::Text("w1 = %.4f;", _state->class_data.current_weights.w1);
		ImGui::EndGroup();
		ImGui::SameLine(separator);
		ImGui::BeginGroup();
		ImGui::Text(" w2 = %.4f;", _state->class_data.current_weights.w2);
		ImGui::EndGroup();
		ImGui::SameLine(2 * separator);
		ImGui::BeginGroup();
		ImGui::Text("w0 = %.4f", _state->class_data.current_weights.w0);
		ImGui::EndGroup();
	}

	ImGui::End();

	if (_state->isPlaying) {
		if (const double currentTime = glfwGetTime();
			currentTime - _state->lastUpdateTime >= 1.0 / _state->playSpeed) {
			_state->iteration++;
			if (_state->iteration >= (int)_state->reg_data.history.size()) {
				_state->iteration = (int)_state->reg_data.history.size() - 1;
				_state->isPlaying = false;
			}
			_state->lastUpdateTime = currentTime;
			}
	}

	_state->reg_data.current_weights = _state->reg_data.history[_state->iteration];
}