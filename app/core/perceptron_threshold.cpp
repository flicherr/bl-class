#include "perceptron.h"

int PerceptronThreshold::predict(double x1, double x2) const
{
	return (w1 * x1 + w2 * x2 - t >= 0.0) ? 1 : -1;
}

void PerceptronThreshold::update(double x1, double x2, int y)
{
	if (int y_hat = predict(x1, x2); y_hat!= y) {
		w1 += lr * y * x1;
		w2 += lr * y * x2;
		// w1 = lr * (y_hat - y) * x1;
		// w2 = lr * (y_hat - y) * x2;
	}
}

Weights PerceptronThreshold::getWeights() const
{
	return { -t, w1, w2 };
}

void PerceptronThreshold::setWeights(const Weights& w)
{
	w1 = w.w1;
	w2 = w.w2;
	t  = -w.w0;
}
