#ifndef APPSTATE_H
#define APPSTATE_H

#include <array>
#include <vector>
#include <imgui.h>
#include "neurons/neurons.h"
#include "samples.h"

struct ClassifierData
{
    std::array<neurons::TypeClassifier, 2> type_class
        = { neurons::Threshold, neurons::Bias };
    std::vector<ClSample> train;
    std::vector<ClSample> test;
    std::array<std::vector<ClWeights>, 2> history;
    std::array<ClWeights, 2> current_weights;
};

struct RegressionData
{
    std::vector<RegSample> train;
    std::vector<RegSample> test;
    std::vector<RegWeights> history;
    RegWeights current_weights{0, 0};
};

struct AppState
{
    ClassifierData class_data;
    RegressionData reg_data;

    int iteration = 0;

    bool isPlaying = false;
    float playSpeed = 1.0f;
    double lastUpdateTime = 0.0;

    ImVec2 offset = ImVec2(0.0f, 0.0f);
    double scale = 44.0;
    bool isDragging = false;
    ImVec2 dragStart;

    enum CurrentShownData {
        Regression,
        ClassificationBias,
        ClassificationThreshold,
    };
    CurrentShownData currentShownData = Regression;
};

#endif //APPSTATE_H