#ifndef UI_MANAGER_H
#define UI_MANAGER_H

#include <memory>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include "appstate.h"

namespace gui
{
class Manager
{
public:
    ~Manager();

    bool init();
    void cleanup();

    void drawing();

private:
    bool init_glfw();
    void init_imgui();

    void set_appstate();

    void draw_canvas() const;
    void draw_toolbar() const;

    ImVec2 worldToScreen(
        double x, double y, ImVec2 origin, ImVec2 size) const;

private:
    GLFWwindow *_window = nullptr;
    std::shared_ptr<AppState> _state;
};
}

#endif //UI_MANAGER_H