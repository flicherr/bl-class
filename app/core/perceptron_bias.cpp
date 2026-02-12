#include "perceptron.h"

int PerceptronBias::predict(double x1, double x2) const
{
    return (w1 * x1 + w2 * x2 + w0 >= 0.0) ? 1 : -1;
}

void PerceptronBias::update(double x1, double x2, int y)
{
    if (int y_hat = predict(x1, x2); y_hat != y) {
        w1 += lr * y * x1;
        w2 += lr * y * x2;
        w0 += lr * y;
        // w1 = lr * (y_hat - y) * x1;
        // w2 = lr * (y_hat - y) * x2;
        // w0 = lr * (y_hat - y);
    }
}

Weights PerceptronBias::getWeights() const
{
    return { w0, w1, w2 };
}

void PerceptronBias::setWeights(const Weights& w)
{
    w0 = w.w0;
    w1 = w.w1;
    w2 = w.w2;
}