#include "neurons.h"

double neurons::Regression::predict(const double &x1) const
{
    return x1 * _weights.w1 + _weights.w0;
}

void neurons::Regression::update(double x1, double y)
{
    auto y_hat = predict(x1);
    // _weights.w1 += _lr * y * x1;
    // _weights.w0 += _lr * y;
    _weights.w1 += _lr * (y - y_hat) * x1;
    _weights.w0 += _lr * (y - y_hat);
}

RegWeights neurons::Regression::getWeights() const
{
    return _weights;
}

void neurons::Regression::setWeights(const RegWeights &w)
{
    _weights = w;
}