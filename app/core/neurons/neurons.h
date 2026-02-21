#ifndef NEURON_H
#define NEURON_H

#include "../weights.h"

namespace neurons
{
class IClassifier
{
public:
    virtual ~IClassifier() = default;

    virtual int predict(double x1, double x2) const = 0;
    virtual void update(double x1, double x2, int y) = 0;

    virtual ClWeights getWeights() const = 0;
    virtual void setWeights(const ClWeights &w) = 0;
};

enum TypeClassifier { Threshold, Bias };

class ClassifierByThreshold final : public IClassifier
{
public:
    int predict(double x1, double x2) const override;
    void update(double x1, double x2, int y) override;
    ClWeights getWeights() const override;
    void setWeights(const ClWeights &w) override;

    double w1 = 0.0;
    double w2 = 0.0;
    double t  = -2.0;
    double lr = 0.1;
};

class ClassifierByBias final : public IClassifier
{
public:
    int predict(double x1, double x2) const override;
    void update(double x1, double x2, int y) override;
    ClWeights getWeights() const override;
    void setWeights(const ClWeights &w) override;

    double w1 = 0.0;
    double w2 = 0.0;
    double w0 = 0.0;
    double lr = 0.1;
};

class Regression
{
public:
    double predict(const double &x1) const;
    void update(double x1, double y);
    RegWeights getWeights() const;
    void setWeights(const RegWeights &w);

private:
    RegWeights _weights;
    double _lr = 0.04;
};
}

#endif //NEURON_H