#include <random>
#include <algorithm>
#include "trainers.h"

void ClassifierTrainer::train(
    neurons::IClassifier &model,
    std::vector<ClSample> data,
    int epochs,
    bool shuffle
) {
    std::vector<ClSample> samples = data;
    std::mt19937 rng{ std::random_device{}() };

    history.clear();
    history.push_back(model.getWeights());

    for (int e = 0; e < epochs; ++e) {
        if (shuffle) {
            std::shuffle(samples.begin(), samples.end(), rng);
        }

        int errors = 0;
        for (const auto& s : samples) {
            if (int pred = model.predict(s.x1, s.x2); pred != s.y) {
                ++errors;
                model.update(s.x1, s.x2, s.y);
                history.push_back(model.getWeights());
            }
        }

        if (errors == 0) break;
    }
}

double ClassifierTrainer::test(
    neurons::IClassifier &model,
    const std::vector<ClSample> &data
) {
    int correct = 0;
    for (const auto& s : data) {
        if (model.predict(s.x1, s.x2) == s.y) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / data.size();
}