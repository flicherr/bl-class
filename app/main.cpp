#include "gui/gui_manager.h"

/* FIXME: transfer the logic of the visualization interface
 * (REMAINS ONLY DRAWING LINE (weight) FOR CLASSIFICATION) from main.cpp in the app/gui
 *
 */
// 	const auto &w = state.current_weights;
// 	if (state.type_preceptron == classifier::TypePerceptron::Bias) {
// 		if (w.w2 != 0.0) {
// 			double x_min = -10.0, x_max = 10.0;
// 			double y1 = -(w.w1 * x_min + w.w0) / w.w2;
// 			double y2 = -(w.w1 * x_max + w.w0) / w.w2;
// 			ImVec2 p1 = worldToScreen(x_min, y1, canvas_pos, canvas_size, scale);
// 			ImVec2 p2 = worldToScreen(x_max, y2, canvas_pos, canvas_size, scale);
// 			draw_list->AddLine(p1, p2, IM_COL32(200, 200, 200, 255), 2.0f);
// 		} else if (w.w1 != 0.0) {
// 			double x = -w.w0 / w.w1;
// 			ImVec2 p1 = worldToScreen(x, -10, canvas_pos, canvas_size, scale);
// 			ImVec2 p2 = worldToScreen(x,  10, canvas_pos, canvas_size, scale);
// 			draw_list->AddLine(p1, p2, IM_COL32(200, 200, 200, 255), 2.0f);
// 		}
// 	} else if (state.type_preceptron == classifier::TypePerceptron::Threshold) {
// 		if (w.w2 != 0.0) {
// 			double x_min = -10.0, x_max = 10.0;
// 			double y1 = -(w.w1 * x_min + w.w0) / w.w2;
// 			double y2 = -(w.w1 * x_max + w.w0) / w.w2;
// 			ImVec2 p1 = worldToScreen(x_min, y1, canvas_pos, canvas_size, scale);
// 			ImVec2 p2 = worldToScreen(x_max, y2, canvas_pos, canvas_size, scale);
// 			draw_list->AddLine(p1, p2, IM_COL32(200, 200, 200, 255), 2.0f);
// 		} else if (w.w1 != 0.0) {
// 			double x = w.w0 / w.w1;
// 			ImVec2 p1 = worldToScreen(x, -10, canvas_pos, canvas_size, scale);
// 			ImVec2 p2 = worldToScreen(x,  10, canvas_pos, canvas_size, scale);
// 			draw_list->AddLine(p1, p2, IM_COL32(200, 200, 200, 255), 2.0f);
// 		}
// 	}

int main()
{
    gui::Manager mgr;
    mgr.init();

    mgr.drawing();

    return 0;
}