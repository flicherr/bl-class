#ifndef PERCEPTRON_H
#define PERCEPTRON_H

struct Weights
{
	double w0;
	double w1;
	double w2;
};

class IPerceptron
{
public:
	virtual ~IPerceptron() = default;

	virtual int predict(double x1, double x2) const = 0;
	virtual void update(double x1, double x2, int y) = 0;

	virtual Weights getWeights() const = 0;
	virtual void setWeights(const Weights &w) = 0;
};

class PerceptronThreshold : public IPerceptron
{
public:
	int predict(double x1, double x2) const override;
	void update(double x1, double x2, int y) override;
	Weights getWeights() const override;
	void setWeights(const Weights &w) override;

	double w1 = 0.0;
	double w2 = 0.0;
	double t  = -2.0;
	double lr = 0.1;
};

class PerceptronBias : public IPerceptron
{
public:
	int predict(double x1, double x2) const override;
	void update(double x1, double x2, int y) override;
	Weights getWeights() const override;
	void setWeights(const Weights &w) override;

	double w0 = 0.0;
	double w1 = 0.0;
	double w2 = 0.0;
	double lr = 0.1;
};

#endif //PERCEPTRON_H