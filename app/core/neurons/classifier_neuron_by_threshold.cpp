#include "neurons.h"

int neurons::ClassifierByThreshold::predict(double x1, double x2) const
{
    return (w1 * x1 + w2 * x2 - t >= 0.0) ? 1 : -1;
}

void neurons::ClassifierByThreshold::update(double x1, double x2, int y)
{
    if (int y_hat = predict(x1, x2); y_hat!= y) {
        // w1 += lr * y * x1;
        // w2 += lr * y * x2;
        w1 += lr * (y - y_hat) * x1;
        w2 += lr * (y - y_hat) * x2;
    }
}

ClWeights neurons::ClassifierByThreshold::getWeights() const
{
    return { w1, w2, -t };
}

void neurons::ClassifierByThreshold::setWeights(const ClWeights &w)
{
    w1 = w.w1;
    w2 = w.w2;
    t  = -w.w0;
}