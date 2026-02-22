#include "gui/gui_manager.h"

int main()
{
    gui::Manager mgr;
    mgr.init();

    mgr.drawing();

    return 0;
}