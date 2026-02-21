#ifndef TRAINERS_H
#define TRAINERS_H

#include <vector>
#include "samples.h"
#include "neurons/neurons.h"

class ClassifierTrainer
{
public:
    void train(
        neurons::IClassifier &model,
        std::vector<ClSample> data,
        int max_epochs,
        bool shuffle = true
    );

    double test(
        neurons::IClassifier& model,
        const std::vector<ClSample>& data
    );

    std::vector<ClWeights> history;
};

class RegressionTrainer
{
public:
    void train(
        neurons::Regression &model,
        std::vector<RegSample> data,
        int max_epochs = 1000,
        double eps = 1e-64,
        bool shuffle = true
    );

    double test(
        neurons::Regression &model,
        const std::vector<RegSample>& data
    );

    double computeMSE(
        neurons::Regression &model,
        const std::vector<RegSample> &data
    );

    std::vector<RegWeights> history;
    std::vector<double> loss_history;
};

#endif //TRAINERS_H