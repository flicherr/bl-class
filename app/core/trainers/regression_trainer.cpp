#include <algorithm>
#include <random>
#include <limits>
#include "trainers.h"

void RegressionTrainer::train(
        neurons::Regression &model,
        std::vector<RegSample> data,
        int max_epochs,
        double eps,
        bool shuffle
) {
    history.clear();
    loss_history.clear();
    history.push_back(model.getWeights());

    std::vector<RegSample> shuffled = data;
    std::mt19937 rng{ std::random_device{}() };

    double prev_loss = std::numeric_limits<double>::infinity();

    for (int epoch = 0; epoch < max_epochs; ++epoch)
    {
        if (shuffle) {
            std::ranges::shuffle(shuffled, rng);
        }

        for (const auto &s : shuffled) {
            model.update(s.x, s.y);
            history.push_back(model.getWeights());
        }

        double loss = computeMSE(model, data);
        loss_history.push_back(loss);

        if (prev_loss - loss < eps) {
            break;
        }

        prev_loss = loss;
    }
}

double RegressionTrainer::test(
    neurons::Regression &model,
    const std::vector<RegSample> &data
) {
    if (data.empty()) {
        return 0.0;
    }

    double sum = 0.0;

    for (const auto& s : data) {
        double y_pred = model.predict(s.x);
        double diff = y_pred - s.y;
        sum += diff * diff;
    }

    return sum / static_cast<double>(data.size());
}

double RegressionTrainer::computeMSE(
    neurons::Regression &model,
    const std::vector<RegSample> &data
) {
    double sum = 0.0;

    for (const auto& s : data)
    {
        double y_pred = model.predict(s.x);
        double diff = y_pred - s.y;
        sum += diff * diff;
    }

    return sum / static_cast<double>(data.size());
}